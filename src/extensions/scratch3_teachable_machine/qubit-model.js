const tf = require("@tensorflow/tfjs");
const tmImage = require("@teachablemachine/image");
const tmPose = require("@teachablemachine/pose");
const tmAudioSpeechCommands = require("@tensorflow-models/speech-commands");

// loadPoseNet is not re-exported from the package entry point, and the only
// public way in - tmPose.load - insists on fetching model.json over HTTP.
const { loadPoseNet } = require("@teachablemachine/pose/dist/custom-posenet");


/**
 * Bump this whenever this file changes. It is printed as soon as the module is
 * evaluated, so the console immediately says which build a deployment is
 * actually running - the difference between "the code is broken" and "the code
 * was never shipped".
 * @type {string}
 */
const QUBIT_MODEL_BUILD = "2026-08-05-debug1";

/**
 * Logging is on by default so a deployed environment can be debugged without a
 * rebuild. Silence it from the console with:
 *   window.__QUBIT_MODEL_DEBUG = false
 * @returns {boolean} whether to emit debug output
 */
const debugEnabled = () =>
    typeof window === "undefined" || window.__QUBIT_MODEL_DEBUG !== false;

const LOG_PREFIX = "[qubit-model]";

const log = (...args) => {
    if (debugEnabled()) {
        console.log(LOG_PREFIX, ...args);
    }
};

const logError = (...args) => {
    // Errors are never silenced - they are the reason anyone is reading this.
    console.error(LOG_PREFIX, ...args);
};

/**
 * Milliseconds since page load, for timing individual steps. performance is not
 * guaranteed to exist in every host the VM runs in.
 * @returns {number} a monotonic-ish timestamp
 */
const now = () =>
    typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();

/**
 * @param {number} startedAt - a timestamp from now()
 * @returns {string} elapsed time, formatted
 */
const since = (startedAt) => `${Math.round(now() - startedAt)}ms`;

log(
    `module loaded, build ${QUBIT_MODEL_BUILD}. ` +
        `Set window.__QUBIT_MODEL_DEBUG = false to silence this.`
);

const QUBIT_HOST_PATTERN = /(^|\.)myqubit\.co$/i;

const IMAGE_SIZE = 224;

/**
 * Alphas that MobileNet v2 checkpoints exist for.
 * @type {Array.<number>}
 */
const VALID_V2_ALPHAS = [0.35, 0.5, 0.75, 1];

/**
 * The `modelMetadata.type` / `metadata.modelType` values the backend emits,
 * mapped onto the model kinds this module knows how to build.
 * @type {object}
 */
const MODEL_TYPES = {
    image: "image",
    "image-classification": "image",
    audio: "audio",
    "audio-classification": "audio",
    pose: "pose",
    "pose-classification": "pose",
    text: "text",
    "text-classification": "text",
};

/**
 * The only audio embedding model there is a browser pipeline for.
 * @type {string}
 */
const AUDIO_EMBEDDING_MODEL = "speech_commands_browser_fft";

/**
 * BrowserFftSpeechCommandRecognizer hardcodes its sample rate and FFT size, so
 * a model trained against different ones would be fed spectrograms that do not
 * line up with anything it saw during training.
 * @type {number}
 */
const BROWSER_FFT_SAMPLE_RATE = 44100;
const BROWSER_FFT_SIZE = 1024;

/**
 * The only pose embedding model there is a browser pipeline for.
 * @type {string}
 */
const POSE_EMBEDDING_MODEL = "posenet_mobilenet_v1";

/**
 * PoseNet emits, per output-grid cell, one heatmap score plus an x and a y
 * offset for each of its 17 keypoints. Teachable Machine's pose embedding is
 * those three planes concatenated and flattened.
 * @type {number}
 */
const POSE_VALUES_PER_CELL = 17 * 3;

/**
 * The defaults @teachablemachine/pose falls back to. Spelled out here so the
 * embedding size can be checked against the settings actually being used.
 * @type {object}
 */
const DEFAULT_POSENET_SETTINGS = {
    architecture: "MobileNetV1",
    outputStride: 16,
    inputResolution: 257,
    multiplier: 0.75,
};

/**
 * Text embeddings come from a transformers.js sentence encoder, which is an
 * ONNX model rather than a tfjs one. Loading it from a CDN at runtime keeps
 * onnxruntime and its wasm binaries out of the bundle entirely.
 * @type {string}
 */
const TRANSFORMERS_CDN_URL =
    "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

/**
 * The sentence encoder to fall back on when the model does not name one.
 * @type {string}
 */
const DEFAULT_TEXT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";


let authTokenProvider = null;


const setAuthTokenProvider = (provider) => {
    log("setAuthTokenProvider called, provider is", typeof provider);
    authTokenProvider = provider;
};


let assetBaseUrlOverride = null;

/**
 * Point asset loading at a specific host, overriding the derived default.
 * @param {?string} baseUrl - a base URL, or null to go back to deriving it
 */
const setAssetBaseUrl = (baseUrl) => {
    log("setAssetBaseUrl called with", baseUrl);
    assetBaseUrlOverride = baseUrl;
};


const resolveAssetBaseUrl = (modelUrl) => {
    if (assetBaseUrlOverride) {
        log("resolveAssetBaseUrl: using override ->", assetBaseUrlOverride);
        return assetBaseUrlOverride;
    }
    const { protocol, host } = new URL(modelUrl);
    const assetHost = host.startsWith("static.") ? host : `static.${host}`;
    const resolved = `${protocol}//${assetHost}/`;
    log(
        `resolveAssetBaseUrl: no override set, derived from model host "${host}" -> ${resolved}`
    );
    return resolved;
};


const featureExtractorCache = {};

/**
 * @param {string} modelArg - the raw string typed into the "use model" block
 * @returns {boolean} true if this should be loaded as a myQubit model
 */
const isQubitModelUrl = (modelArg) => {
    try {
        const { hostname } = new URL(modelArg);
        const matched = QUBIT_HOST_PATTERN.test(hostname);
        log(
            `isQubitModelUrl("${modelArg}"): hostname "${hostname}" -> ${matched}`
        );
        return matched;
    } catch (e) {
        // Not an absolute URL, so it is a bare Teachable Machine model id.
        log(
            `isQubitModelUrl("${modelArg}"): not an absolute URL, ` +
                `treating as a Teachable Machine model id`
        );
        return false;
    }
};

/**
 * @returns {object} fetch options carrying whatever credentials we have
 */
const buildRequestInit = () => {
    const init = { credentials: "include" };
    const token = authTokenProvider ? authTokenProvider() : null;
    if (token) {
        init.headers = { Authorization: `Bearer ${token}` };
    }
    log("buildRequestInit:", {
        credentials: init.credentials,
        hasAuthTokenProvider: Boolean(authTokenProvider),
        hasToken: Boolean(token),
    });
    if (!authTokenProvider) {
        log(
            "buildRequestInit: no auth token provider registered - requests " +
                "rely entirely on cookies, which are not sent cross-site " +
                "unless the session cookie is SameSite=None; Secure."
        );
    }
    return init;
};


const parseEmbeddingModel = (embeddingModel) => {
    const match = /^mobilenet_v(\d+)_([\d.]+)$/i.exec(embeddingModel || "");
    if (!match) {
        throw new Error(
            `Unsupported embedding model "${embeddingModel}". ` +
                `Expected something like "mobilenet_v2_1.0".`
        );
    }
    const version = Number(match[1]);
    const alpha = Number(match[2]);
    if (version !== 2 || VALID_V2_ALPHAS.indexOf(alpha) === -1) {
        throw new Error(
            `Unsupported embedding model "${embeddingModel}". ` +
                `Only MobileNet v2 with alpha ${VALID_V2_ALPHAS.join("/")} is available.`
        );
    }
    return { version, alpha };
};

/**
 * @param {object} spec - {version, alpha}
 * @returns {Promise} resolves to a tf.Sequential producing [1, 1280] embeddings
 */
const getFeatureExtractor = (spec) => {
    const key = `v${spec.version}_a${spec.alpha}`;
    if (!featureExtractorCache[key]) {
        // Truncates MobileNet at out_relu and global-average-pools it, which is
        // exactly how the embeddings in the training set were produced.
        featureExtractorCache[key] = tmImage
            .loadTruncatedMobileNet(spec)
            .catch((e) => {
                // Do not cache a rejected promise, or every later attempt fails.
                delete featureExtractorCache[key];
                throw e;
            });
    }
    return featureExtractorCache[key];
};


const cropToSquare = (image) => {
    const width =
        image instanceof HTMLVideoElement ? image.videoWidth : image.width;
    const height =
        image instanceof HTMLVideoElement ? image.videoHeight : image.height;

    const scale = IMAGE_SIZE / Math.min(width, height);
    const scaledWidth = Math.ceil(width * scale);
    const scaledHeight = Math.ceil(height * scale);

    const canvas = document.createElement("canvas");
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    canvas
        .getContext("2d")
        .drawImage(
            image,
            ~~((scaledWidth - IMAGE_SIZE) / 2) * -1,
            ~~((scaledHeight - IMAGE_SIZE) / 2) * -1,
            scaledWidth,
            scaledHeight
        );
    return canvas;
};


const canvasToBatch = (canvas) =>
    tf.tidy(() =>
        tf.browser
            .fromPixels(canvas)
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(127))
            .sub(tf.scalar(1))
    );

class QubitImageModel {
    /**
     * @param {tf.LayersModel} head - the trained classifier head
     * @param {tf.LayersModel} featureExtractor - truncated MobileNet
     * @param {Array.<string>} labels - class names, in output order
     */
    constructor(head, featureExtractor, labels) {
        this.head = head;
        this.featureExtractor = featureExtractor;
        this.labels = labels;
    }

    /**
     * @returns {Array.<string>} the class names this model predicts
     */
    getClassLabels() {
        return this.labels;
    }

    /**
     * @param {ImageBitmap|HTMLCanvasElement} image - the frame to classify
     * @returns {Promise<Array.<object>>} [{className, probability}, ...]
     */
    async predict(image) {
        // This runs on every frame, so everything intermediate is disposed.
        const logits = tf.tidy(() => {
            const batch = canvasToBatch(cropToSquare(image));
            const embedding = this.featureExtractor.predict(batch);
            return this.head.predict(embedding);
        });
        const probabilities = await logits.data();
        logits.dispose();

        return this.labels.map((className, index) => ({
            className,
            probability: probabilities[index],
        }));
    }

    /**
     * Releases the head. The feature extractor is shared, so it is left alone.
     */
    dispose() {
        this.head.dispose();
    }
}


const fetchWeights = async (trainingConfig, origin) => {
    log("fetchWeights: START");

    const manifest = trainingConfig.modelJson.weightsManifest;
    if (!manifest || !manifest[0] || !manifest[0].weights) {
        logError(
            "fetchWeights: weightsManifest is missing or malformed:",
            manifest
        );
        throw new Error("Model response has a missing or malformed weightsManifest.");
    }
    const weightSpecs = manifest[0].weights;
    log(
        `fetchWeights: manifest has ${weightSpecs.length} tensors, ` +
            `paths = ${JSON.stringify(manifest[0].paths)}`
    );

    // weightsPath is origin-relative ("ml-models/..."), not relative to the
    // model endpoint - resolving against the latter would duplicate the prefix.
    const weightsUrl = new URL(trainingConfig.weightsPath, origin).href;
    log("fetchWeights: weightsPath (raw) =", trainingConfig.weightsPath);
    log("fetchWeights: origin            =", origin);
    log("fetchWeights: ABOUT TO FETCH    =", weightsUrl);

    const startedAt = now();
    let response;
    try {
        response = await fetch(weightsUrl, buildRequestInit());
    } catch (e) {
        // A network-layer failure (CORS rejection, DNS, blocked by CSP) never
        // produces a response object, so it must be reported separately or it
        // looks identical to a bad status code.
        logError(
            `fetchWeights: fetch REJECTED after ${since(startedAt)} for ${weightsUrl}`,
            e
        );
        logError(
            "fetchWeights: a rejected fetch (rather than a bad status) usually " +
                "means CORS, DNS or a Content-Security-Policy connect-src block. " +
                "Check the Network tab entry for this URL and the CSP on the page."
        );
        throw e;
    }

    log(
        `fetchWeights: response in ${since(startedAt)} - ` +
            `status ${response.status}, ok=${response.ok}, type=${response.type}`
    );
    if (!response.ok) {
        logError(
            `fetchWeights: bad status ${response.status} from ${weightsUrl}`
        );
        throw new Error(
            `Could not fetch model weights (HTTP ${response.status}) from ${weightsUrl}`
        );
    }
    const weightData = await response.arrayBuffer();
    log(`fetchWeights: downloaded ${weightData.byteLength} bytes`);

    // A truncated download or an error page would otherwise fail deep inside
    // tfjs with an unreadable message.
    const expectedBytes =
        weightSpecs.reduce(
            (total, spec) =>
                total + spec.shape.reduce((a, b) => a * b, 1),
            0
        ) * 4;
    if (weightData.byteLength !== expectedBytes) {
        logError(
            `fetchWeights: SIZE MISMATCH - got ${weightData.byteLength}, ` +
                `expected ${expectedBytes}. A response this size is often an ` +
                `HTML error/login page rather than a .bin.`
        );
        throw new Error(
            `Model weights are ${weightData.byteLength} bytes, expected ${expectedBytes}.`
        );
    }

    log("fetchWeights: DONE, size matches manifest");
    return { weightSpecs, weightData };
};


const loadHead = async (trainingConfig, origin) => {
    log("loadHead: START");
    const { weightSpecs, weightData } = await fetchWeights(
        trainingConfig,
        origin
    );
    log("loadHead: handing artifacts to tf.loadLayersModel");

    return tf.loadLayersModel(
        tf.io.fromMemory({
            modelTopology: trainingConfig.modelJson.modelTopology,
            weightSpecs: weightSpecs,
            weightData: weightData,
            format: trainingConfig.modelJson.format,
            generatedBy: trainingConfig.modelJson.generatedBy,
            convertedBy: trainingConfig.modelJson.convertedBy,
        })
    ).then((head) => {
        log(
            "loadHead: DONE, head input shape =",
            JSON.stringify(head.inputs[0].shape)
        );
        return head;
    });
};

/**
 * @param {object} metadata - trainingConfig.metadata for an audio model
 */
const checkAudioSettings = (metadata) => {
    log("checkAudioSettings: embeddingModel =", metadata.embeddingModel);
    log(
        "checkAudioSettings: audioSettings =",
        JSON.stringify(metadata.audioSettings)
    );
    if (
        metadata.embeddingModel &&
        metadata.embeddingModel !== AUDIO_EMBEDDING_MODEL
    ) {
        throw new Error(
            `Unsupported embedding model "${metadata.embeddingModel}". ` +
                `Only "${AUDIO_EMBEDDING_MODEL}" is available.`
        );
    }

    // The frame count and frequency bin count do not need checking: the
    // recognizer reads those straight off the model's input shape.
    const audioSettings = metadata.audioSettings || {};
    if (
        audioSettings.sampleRate &&
        audioSettings.sampleRate !== BROWSER_FFT_SAMPLE_RATE
    ) {
        throw new Error(
            `Model was trained at ${audioSettings.sampleRate} Hz but the browser ` +
                `recognizer always samples at ${BROWSER_FFT_SAMPLE_RATE} Hz.`
        );
    }
    if (audioSettings.fftSize && audioSettings.fftSize !== BROWSER_FFT_SIZE) {
        throw new Error(
            `Model was trained with an FFT size of ${audioSettings.fftSize} but the ` +
                `browser recognizer always uses ${BROWSER_FFT_SIZE}.`
        );
    }
    log("checkAudioSettings: OK");
};

/**
 * @param {object} trainingConfig - the validated trainingConfig
 * @param {object} metadata - trainingConfig.metadata
 * @param {Array.<string>} labels - class names, in output order
 * @param {string} origin - base URL to resolve weightsPath against
 * @returns {Promise} resolves to a loaded SpeechCommandRecognizer
 */
const loadAudioRecognizer = async (
    trainingConfig,
    metadata,
    labels,
    origin
) => {
    log("loadAudioRecognizer: START");
    checkAudioSettings(metadata);

    const { weightSpecs, weightData } = await fetchWeights(
        trainingConfig,
        origin
    );

    // Unlike the image models, the exported topology is the whole network -
    // the frozen speech-commands convnet with the trained head grafted on -
    // so there is nothing to load alongside it. Handing the recognizer the
    // artifacts directly keeps it from going out to model.json/metadata.json,
    // which myQubit does not serve as static files.
    const recognizer = tmAudioSpeechCommands.create(
        "BROWSER_FFT",
        undefined,
        {
            modelTopology: trainingConfig.modelJson.modelTopology,
            weightSpecs: weightSpecs,
            weightData: weightData,
        },
        {
            tfjsSpeechCommandsVersion: metadata.packageVersion || "0.5.4",
            modelName: metadata.modelName,
            wordLabels: labels,
        }
    );
    log(
        "loadAudioRecognizer: recognizer created, calling ensureModelLoaded(). " +
            "If this is the last line you see, speech-commands is trying to " +
            "reach its own model host and failing."
    );
    const startedAt = now();
    try {
        await recognizer.ensureModelLoaded();
    } catch (e) {
        logError(
            `loadAudioRecognizer: ensureModelLoaded FAILED after ${since(startedAt)}`,
            e
        );
        throw e;
    }
    log(`loadAudioRecognizer: DONE in ${since(startedAt)}`);
    return recognizer;
};

const posenetCache = {};

/**
 * @param {object} metadata - trainingConfig.metadata for a pose model
 * @returns {object} the PoseNet settings the model was trained against
 */
const resolvePosenetSettings = (metadata) => {
    const settings =
        (metadata.modelSettings && metadata.modelSettings.posenet) || {};
    return {
        architecture:
            settings.architecture || DEFAULT_POSENET_SETTINGS.architecture,
        outputStride:
            settings.outputStride || DEFAULT_POSENET_SETTINGS.outputStride,
        inputResolution:
            settings.inputResolution ||
            DEFAULT_POSENET_SETTINGS.inputResolution,
        multiplier: settings.multiplier || DEFAULT_POSENET_SETTINGS.multiplier,
    };
};

/**
 * @param {object} settings - resolved PoseNet settings
 * @returns {?number} the length of the embedding those settings produce, or
 *   null if it cannot be worked out
 */
const posenetOutputDim = (settings) => {
    const { inputResolution, outputStride } = settings;
    if (
        typeof inputResolution !== "number" ||
        typeof outputStride !== "number"
    ) {
        // inputResolution may also be given as {width, height}, which the
        // trainer does not use.
        return null;
    }
    const gridSize = (inputResolution - 1) / outputStride + 1;
    if (!Number.isInteger(gridSize)) {
        return null;
    }
    return gridSize * gridSize * POSE_VALUES_PER_CELL;
};

/**
 * @param {object} settings - resolved PoseNet settings
 * @returns {Promise} resolves to a posenet.PoseNet
 */
const getPosenet = (settings) => {
    const key = [
        settings.architecture,
        settings.outputStride,
        settings.inputResolution,
        settings.multiplier,
    ].join("_");
    if (!posenetCache[key]) {
        log(
            `getPosenet: cache miss for "${key}", downloading PoseNet ` +
                `(this goes to Google's model host, not myQubit)`
        );
        const startedAt = now();
        posenetCache[key] = loadPoseNet({ posenet: settings })
            .then((net) => {
                log(`getPosenet: loaded in ${since(startedAt)}`);
                return net;
            })
            .catch((e) => {
                // Do not cache a rejected promise, or every later attempt fails.
                logError(
                    `getPosenet: FAILED after ${since(startedAt)} for "${key}"`,
                    e
                );
                delete posenetCache[key];
                throw e;
            });
    } else {
        log(`getPosenet: cache hit for "${key}"`);
    }
    return posenetCache[key];
};

/**
 * @param {object} trainingConfig - the validated trainingConfig
 * @param {object} metadata - trainingConfig.metadata
 * @param {Array.<string>} labels - class names, in output order
 * @param {string} origin - base URL to resolve weightsPath against
 * @returns {Promise} resolves to a tmPose.CustomPoseNet
 */
const loadPoseModel = async (trainingConfig, metadata, labels, origin) => {
    if (
        metadata.embeddingModel &&
        metadata.embeddingModel !== POSE_EMBEDDING_MODEL
    ) {
        throw new Error(
            `Unsupported embedding model "${metadata.embeddingModel}". ` +
                `Only "${POSE_EMBEDDING_MODEL}" is available.`
        );
    }

    const settings = resolvePosenetSettings(metadata);
    log("loadPoseModel: posenet settings =", JSON.stringify(settings));

    // The head and PoseNet are independent, so fetch in parallel.
    log("loadPoseModel: loading head + PoseNet in parallel");
    const [head, posenetModel] = await Promise.all([
        loadHead(trainingConfig, origin),
        getPosenet(settings),
    ]);
    log("loadPoseModel: both head and PoseNet resolved");

    // If these disagree the head was trained against differently configured
    // PoseNet and predictions would be confident nonsense rather than an
    // obvious failure.
    const expectedDim = head.inputs[0].shape[1];
    const actualDim = posenetOutputDim(settings);
    log(
        `loadPoseModel: head expects ${expectedDim}-d, PoseNet produces ${actualDim}-d`
    );
    if (actualDim !== null && expectedDim !== actualDim) {
        head.dispose();
        throw new Error(
            `Model expects ${expectedDim}-d pose embeddings but PoseNet at ` +
                `resolution ${settings.inputResolution}/stride ${settings.outputStride} ` +
                `produces ${actualDim}-d.`
        );
    }

    log("loadPoseModel: DONE");
    // CustomPoseNet is exactly the shape the extension's pose branch expects:
    // estimatePose() then predict() on the embedding it returns.
    return new tmPose.CustomPoseNet(head, posenetModel, {
        labels: labels,
        modelName: metadata.modelName,
        modelSettings: { posenet: settings },
    });
};

let transformersPromise = null;
let transformersUrlOverride = null;

/**
 * Load transformers.js from somewhere other than jsDelivr - a self-hosted copy,
 * for deployments where the CDN is unreachable.
 * @param {?string} url - a module URL, or null to go back to the CDN
 */
const setTransformersUrl = (url) => {
    transformersUrlOverride = url;
    transformersPromise = null;
};

/**
 * @returns {Promise} resolves to the transformers.js module
 */
const getTransformers = () => {
    if (!transformersPromise) {
        const url = transformersUrlOverride || TRANSFORMERS_CDN_URL;
        log(
            `getTransformers: importing transformers.js from ${url} ` +
                `(${transformersUrlOverride ? "override" : "default CDN"})`
        );
        log(
            "getTransformers: this is a cross-origin dynamic import - a strict " +
                "Content-Security-Policy script-src will block it here."
        );
        const startedAt = now();
        // Built via Function so webpack leaves the import alone instead of
        // trying to resolve the URL at build time.
        // eslint-disable-next-line no-new-func
        const importTransformers = new Function(`return import("${url}")`);
        transformersPromise = Promise.resolve()
            .then(importTransformers)
            .then((module) => {
                log(`getTransformers: imported in ${since(startedAt)}`);
                const env = module.env;
                // There is no local model directory to search, and the wasm
                // backend is unreliable with threads in some browsers.
                env.allowLocalModels = false;
                env.allowRemoteModels = true;
                // Caching the encoder matters - it is a multi-megabyte
                // download - but the Cache API is not everywhere.
                env.useBrowserCache = typeof caches !== "undefined";
                if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
                    env.backends.onnx.wasm.numThreads = 1;
                }
                log("getTransformers: env configured, module ready");
                return module;
            })
            .catch((e) => {
                // Do not cache a rejected promise, or every later attempt fails.
                logError(
                    `getTransformers: FAILED after ${since(startedAt)} loading ${url}`,
                    e
                );
                transformersPromise = null;
                throw e;
            });
    } else {
        log("getTransformers: reusing cached module promise");
    }
    return transformersPromise;
};

const embedderCache = {};

/**
 * @param {string} embeddingModel - a transformers.js model id
 * @returns {Promise} resolves to a feature-extraction pipeline
 */
const getTextEmbedder = (embeddingModel) => {
    if (!embedderCache[embeddingModel]) {
        log(
            `getTextEmbedder: building pipeline for "${embeddingModel}" ` +
                `(weights download comes from the Hugging Face CDN)`
        );
        const startedAt = now();
        embedderCache[embeddingModel] = getTransformers()
            .then((module) =>
                module.pipeline("feature-extraction", embeddingModel)
            )
            .then((pipe) => {
                log(`getTextEmbedder: pipeline ready in ${since(startedAt)}`);
                return pipe;
            })
            .catch((e) => {
                logError(
                    `getTextEmbedder: FAILED after ${since(startedAt)} for ` +
                        `"${embeddingModel}"`,
                    e
                );
                delete embedderCache[embeddingModel];
                throw e;
            });
    } else {
        log(`getTextEmbedder: cache hit for "${embeddingModel}"`);
    }
    return embedderCache[embeddingModel];
};

class QubitTextModel {
    /**
     * @param {tf.LayersModel} head - the trained classifier head
     * @param {Function} embedder - a transformers.js feature-extraction pipeline
     * @param {Array.<string>} labels - class names, in output order
     */
    constructor(head, embedder, labels) {
        this.head = head;
        this.embedder = embedder;
        this.labels = labels;
    }

    /**
     * @returns {Array.<string>} the class names this model predicts
     */
    getClassLabels() {
        return this.labels;
    }

    /**
     * @param {string} text - the sentence to classify
     * @returns {Promise<Array.<object>>} [{className, probability}, ...]
     */
    async predict(text) {
        // Mean pooling followed by L2 normalisation is how sentence-transformers
        // turns per-token output into the one vector the head was trained on.
        const output = await this.embedder(String(text), {
            pooling: "mean",
            normalize: true,
        });

        const logits = tf.tidy(() => {
            const embedding = tf.tensor2d(Array.from(output.data), [
                1,
                output.data.length,
            ]);
            return this.head.predict(embedding);
        });
        const probabilities = await logits.data();
        logits.dispose();

        return this.labels.map((className, index) => ({
            className,
            probability: probabilities[index],
        }));
    }

    /**
     * Releases the head. The embedder is shared, so it is left alone.
     */
    dispose() {
        this.head.dispose();
    }
}

/**
 * @param {object} trainingConfig - the validated trainingConfig
 * @param {object} metadata - trainingConfig.metadata
 * @param {Array.<string>} labels - class names, in output order
 * @param {string} origin - base URL to resolve weightsPath against
 * @returns {Promise} resolves to a QubitTextModel
 */
const loadTextModel = async (trainingConfig, metadata, labels, origin) => {
    const embeddingModel =
        metadata.embeddingModel || DEFAULT_TEXT_EMBEDDING_MODEL;
    log("loadTextModel: START, embeddingModel =", embeddingModel);

    // The head and the sentence encoder are independent, so fetch in parallel.
    log("loadTextModel: loading head + sentence encoder in parallel");
    const [head, embedder] = await Promise.all([
        loadHead(trainingConfig, origin),
        getTextEmbedder(embeddingModel),
    ]);
    log("loadTextModel: both head and encoder resolved");

    // Embedding once up front both warms the pipeline and pins down the
    // encoder's real output size. If it disagrees with the head, the head was
    // trained against a different encoder and predictions would be confident
    // nonsense rather than an obvious failure.
    const expectedDim = head.inputs[0].shape[1];
    log("loadTextModel: running probe embedding to confirm encoder output size");
    const probe = await embedder("probe", {
        pooling: "mean",
        normalize: true,
    });
    log(
        `loadTextModel: head expects ${expectedDim}-d, encoder produced ` +
            `${probe.data.length}-d`
    );
    if (probe.data.length !== expectedDim) {
        head.dispose();
        throw new Error(
            `Model expects ${expectedDim}-d text embeddings but ` +
                `${embeddingModel} produces ${probe.data.length}-d.`
        );
    }

    log("loadTextModel: DONE");
    return new QubitTextModel(head, embedder, labels);
};

/**
 * @param {object} payload - the parsed myQubit model response
 * @returns {object} the trainingConfig, validated
 */
const validatePayload = (payload) => {
    log("validatePayload: START");
    log("validatePayload: top-level keys =", Object.keys(payload || {}));

    const trainingConfig =
        payload && payload.trainingData && payload.trainingData.trainingConfig;
    if (!trainingConfig) {
        logError(
            "validatePayload: trainingData.trainingConfig missing. " +
                "trainingData keys =",
            Object.keys((payload && payload.trainingData) || {})
        );
        throw new Error("Model response has no trainingData.trainingConfig.");
    }
    log("validatePayload: trainingConfig keys =", Object.keys(trainingConfig));

    if (!trainingConfig.modelJson || !trainingConfig.modelJson.modelTopology) {
        logError(
            "validatePayload: modelJson.modelTopology missing. modelJson keys =",
            Object.keys(trainingConfig.modelJson || {})
        );
        throw new Error("Model response has no modelJson.modelTopology.");
    }
    if (!trainingConfig.weightsPath) {
        logError("validatePayload: weightsPath missing");
        throw new Error("Model response has no weightsPath.");
    }
    log("validatePayload: weightsPath =", trainingConfig.weightsPath);

    const metadata = trainingConfig.metadata || {};
    const rawModelType =
        metadata.modelType ||
        (payload.modelMetadata && payload.modelMetadata.type);
    const modelType = MODEL_TYPES[rawModelType];
    log(
        `validatePayload: metadata.modelType = ${JSON.stringify(metadata.modelType)}, ` +
            `modelMetadata.type = ${JSON.stringify(
                payload.modelMetadata && payload.modelMetadata.type
            )}`
    );
    log(
        `validatePayload: raw type "${rawModelType}" -> mapped "${modelType}". ` +
            `Known keys: ${Object.keys(MODEL_TYPES).join(", ")}`
    );
    if (!modelType) {
        logError(
            `validatePayload: UNRECOGNISED MODEL TYPE "${rawModelType}". ` +
                `The lookup is case-sensitive and hyphenated. If this build is ` +
                `older than audio/pose/text support, that is the real problem.`
        );
        throw new Error(
            `Model type "${rawModelType}" is not supported yet - only image, audio, pose and text classification are.`
        );
    }

    const labels = metadata.labels || trainingConfig.classLabels;
    if (!Array.isArray(labels) || labels.length === 0) {
        logError("validatePayload: no class labels found");
        throw new Error("Model response has no class labels.");
    }
    log(`validatePayload: DONE - ${labels.length} labels:`, labels);

    return { trainingConfig, metadata, labels, modelType };
};


/**
 * @param {string} modelUrl - a myQubit ml-models endpoint
 * @returns {Promise<object>} resolves to {model, type}, where type is "image",
 *   "audio", "pose" or "text". An audio model is a speech-commands recognizer
 *   that still needs `listen()` called on it.
 */
const loadQubitModel = async (modelUrl) => {
    const startedAt = now();
    log("=".repeat(60));
    log(`loadQubitModel: START build ${QUBIT_MODEL_BUILD}`);
    log("loadQubitModel: modelUrl =", modelUrl);
    log(
        "loadQubitModel: page origin =",
        typeof location === "undefined" ? "(no location)" : location.origin
    );

    let response;
    try {
        response = await fetch(modelUrl, buildRequestInit());
    } catch (e) {
        logError("loadQubitModel: model fetch REJECTED (network/CORS/CSP)", e);
        throw e;
    }
    log(
        `loadQubitModel: model response status ${response.status}, ` +
            `ok=${response.ok}, type=${response.type}`
    );
    if (!response.ok) {
        throw new Error(
            `Could not fetch model (HTTP ${response.status}) from ${modelUrl}`
        );
    }
    const payload = await response.json();
    log("loadQubitModel: payload parsed");

    const { trainingConfig, metadata, labels, modelType } =
        validatePayload(payload);
    const origin = resolveAssetBaseUrl(modelUrl);
    log(`loadQubitModel: TAKING THE "${modelType}" BRANCH`);

    if (modelType === "audio") {
        const recognizer = await loadAudioRecognizer(
            trainingConfig,
            metadata,
            labels,
            origin
        );
        log(`loadQubitModel: SUCCESS (audio) in ${since(startedAt)}`);
        return { model: recognizer, type: modelType };
    }

    if (modelType === "pose") {
        const poseModel = await loadPoseModel(
            trainingConfig,
            metadata,
            labels,
            origin
        );
        log(`loadQubitModel: SUCCESS (pose) in ${since(startedAt)}`);
        return { model: poseModel, type: modelType };
    }

    if (modelType === "text") {
        const textModel = await loadTextModel(
            trainingConfig,
            metadata,
            labels,
            origin
        );
        log(`loadQubitModel: SUCCESS (text) in ${since(startedAt)}`);
        return { model: textModel, type: modelType };
    }

    log("loadQubitModel: image branch, embeddingModel =", metadata.embeddingModel);
    // The head and the feature extractor are independent, so fetch in parallel.
    const [head, featureExtractor] = await Promise.all([
        loadHead(trainingConfig, origin),
        getFeatureExtractor(parseEmbeddingModel(metadata.embeddingModel)),
    ]);

    // If these disagree the head was trained against a different extractor and
    // predictions would be confident nonsense rather than an obvious failure.
    const expectedDim = head.inputs[0].shape[1];
    const actualDim =
        featureExtractor.outputs[0].shape[
            featureExtractor.outputs[0].shape.length - 1
        ];
    log(
        `loadQubitModel: head expects ${expectedDim}-d, extractor produces ${actualDim}-d`
    );
    if (expectedDim !== actualDim) {
        head.dispose();
        throw new Error(
            `Model expects ${expectedDim}-d embeddings but ${metadata.embeddingModel} produces ${actualDim}-d.`
        );
    }

    log(`loadQubitModel: SUCCESS (image) in ${since(startedAt)}`);
    return {
        model: new QubitImageModel(head, featureExtractor, labels),
        type: modelType,
    };
};

// Hung off window as well as exported, so the build a deployment is running can
// be checked from the console without digging through the bundle.
if (typeof window !== "undefined") {
    window.__QUBIT_MODEL_BUILD = QUBIT_MODEL_BUILD;
}

module.exports = {
    QUBIT_MODEL_BUILD,
    isQubitModelUrl,
    loadQubitModel,
    setAuthTokenProvider,
    setAssetBaseUrl,
    setTransformersUrl,
    resolveAssetBaseUrl,
    QubitImageModel,
    QubitTextModel,
};
