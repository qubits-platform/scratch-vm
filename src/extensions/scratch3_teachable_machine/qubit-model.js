const tf = require("@tensorflow/tfjs");
const tmImage = require("@teachablemachine/image");


const QUBIT_HOST_PATTERN = /(^|\.)myqubit\.co$/i;

const IMAGE_SIZE = 224;

/**
 * Alphas that MobileNet v2 checkpoints exist for.
 * @type {Array.<number>}
 */
const VALID_V2_ALPHAS = [0.35, 0.5, 0.75, 1];


let authTokenProvider = null;


const setAuthTokenProvider = (provider) => {
    authTokenProvider = provider;
};


let assetBaseUrlOverride = null;

/**
 * Point asset loading at a specific host, overriding the derived default.
 * @param {?string} baseUrl - a base URL, or null to go back to deriving it
 */
const setAssetBaseUrl = (baseUrl) => {
    assetBaseUrlOverride = baseUrl;
};


const resolveAssetBaseUrl = (modelUrl) => {
    if (assetBaseUrlOverride) {
        return assetBaseUrlOverride;
    }
    const { protocol, host } = new URL(modelUrl);
    const assetHost = host.startsWith("static.") ? host : `static.${host}`;
    return `${protocol}//${assetHost}/`;
};


const featureExtractorCache = {};

/**
 * @param {string} modelArg - the raw string typed into the "use model" block
 * @returns {boolean} true if this should be loaded as a myQubit model
 */
const isQubitModelUrl = (modelArg) => {
    try {
        return QUBIT_HOST_PATTERN.test(new URL(modelArg).hostname);
    } catch (e) {
        // Not an absolute URL, so it is a bare Teachable Machine model id.
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


const loadHead = async (trainingConfig, origin) => {
    const manifest = trainingConfig.modelJson.weightsManifest;
    const weightSpecs = manifest[0].weights;

    // weightsPath is origin-relative ("ml-models/..."), not relative to the
    // model endpoint - resolving against the latter would duplicate the prefix.
    const weightsUrl = new URL(trainingConfig.weightsPath, origin).href;
    const response = await fetch(weightsUrl, buildRequestInit());
    if (!response.ok) {
        throw new Error(
            `Could not fetch model weights (HTTP ${response.status}) from ${weightsUrl}`
        );
    }
    const weightData = await response.arrayBuffer();

    // A truncated download or an error page would otherwise fail deep inside
    // tfjs with an unreadable message.
    const expectedBytes =
        weightSpecs.reduce(
            (total, spec) =>
                total + spec.shape.reduce((a, b) => a * b, 1),
            0
        ) * 4;
    if (weightData.byteLength !== expectedBytes) {
        throw new Error(
            `Model weights are ${weightData.byteLength} bytes, expected ${expectedBytes}.`
        );
    }

    return tf.loadLayersModel(
        tf.io.fromMemory({
            modelTopology: trainingConfig.modelJson.modelTopology,
            weightSpecs: weightSpecs,
            weightData: weightData,
            format: trainingConfig.modelJson.format,
            generatedBy: trainingConfig.modelJson.generatedBy,
            convertedBy: trainingConfig.modelJson.convertedBy,
        })
    );
};

/**
 * @param {object} payload - the parsed myQubit model response
 * @returns {object} the trainingConfig, validated
 */
const validatePayload = (payload) => {
    const trainingConfig =
        payload && payload.trainingData && payload.trainingData.trainingConfig;
    if (!trainingConfig) {
        throw new Error("Model response has no trainingData.trainingConfig.");
    }
    if (!trainingConfig.modelJson || !trainingConfig.modelJson.modelTopology) {
        throw new Error("Model response has no modelJson.modelTopology.");
    }
    if (!trainingConfig.weightsPath) {
        throw new Error("Model response has no weightsPath.");
    }

    const metadata = trainingConfig.metadata || {};
    const modelType =
        metadata.modelType ||
        (payload.modelMetadata && payload.modelMetadata.type);
    if (modelType !== "image-classification" && modelType !== "image") {
        throw new Error(
            `Model type "${modelType}" is not supported yet - only image classification is.`
        );
    }

    const labels = metadata.labels || trainingConfig.classLabels;
    if (!Array.isArray(labels) || labels.length === 0) {
        throw new Error("Model response has no class labels.");
    }

    return { trainingConfig, metadata, labels };
};


const loadQubitModel = async (modelUrl) => {
    const response = await fetch(modelUrl, buildRequestInit());
    if (!response.ok) {
        throw new Error(
            `Could not fetch model (HTTP ${response.status}) from ${modelUrl}`
        );
    }
    const payload = await response.json();

    const { trainingConfig, metadata, labels } = validatePayload(payload);

    // The head and the feature extractor are independent, so fetch in parallel.
    const [head, featureExtractor] = await Promise.all([
        loadHead(trainingConfig, resolveAssetBaseUrl(modelUrl)),
        getFeatureExtractor(parseEmbeddingModel(metadata.embeddingModel)),
    ]);

    // If these disagree the head was trained against a different extractor and
    // predictions would be confident nonsense rather than an obvious failure.
    const expectedDim = head.inputs[0].shape[1];
    const actualDim =
        featureExtractor.outputs[0].shape[
            featureExtractor.outputs[0].shape.length - 1
        ];
    if (expectedDim !== actualDim) {
        head.dispose();
        throw new Error(
            `Model expects ${expectedDim}-d embeddings but ${metadata.embeddingModel} produces ${actualDim}-d.`
        );
    }

    return new QubitImageModel(head, featureExtractor, labels);
};

module.exports = {
    isQubitModelUrl,
    loadQubitModel,
    setAuthTokenProvider,
    setAssetBaseUrl,
    resolveAssetBaseUrl,
    QubitImageModel,
};
