"use strict";

const MAX_FILE_BYTES = 20_000_000;
const MAX_IMAGE_PIXELS = 25_000_000;
const MIN_IMAGE_DIMENSION = 32;
const IMAGE_SIZE = 224;
const MODEL_URL = "models/tfjs/model.json";
const METADATA_URL = "models/model_metadata.json";

const elements = {
  input: document.querySelector("#image-input"),
  dropZone: document.querySelector("#drop-zone"),
  error: document.querySelector("#upload-error"),
  previewCard: document.querySelector("#preview-card"),
  preview: document.querySelector("#image-preview"),
  fileName: document.querySelector("#file-name"),
  remove: document.querySelector("#remove-image"),
  classify: document.querySelector("#classify-button"),
  classifyLabel: document.querySelector("#classify-label"),
  resultEmpty: document.querySelector("#result-empty"),
  resultContent: document.querySelector("#result-content"),
  resultName: document.querySelector("#prediction-name"),
  resultScore: document.querySelector("#prediction-score"),
  topList: document.querySelector("#top-list"),
  modelMessage: document.querySelector("#model-message"),
  supported: document.querySelector("#supported-classes"),
  modelClasses: document.querySelector("#model-class-list"),
  historySection: document.querySelector("#history-section"),
  historyBody: document.querySelector("#history-body"),
};

let selectedFile = null;
let selectedImage = null;
let predictor = null;
let modelLoadPromise = null;
let modelMetadata = null;
const predictionHistory = [];

function showError(message) {
  elements.error.textContent = message;
  elements.error.hidden = false;
}

function clearError() {
  elements.error.textContent = "";
  elements.error.hidden = true;
}

function setSelectedFile(file) {
  if (!file) return;
  resetSelectedFile();
  clearError();

  const extension = file.name.toLowerCase().split(".").pop();
  if (!["jpg", "jpeg", "png"].includes(extension)) {
    showError("Unsupported file type. Choose a JPG, JPEG, or PNG image.");
    return;
  }
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    showError("This file is not a supported JPEG or PNG image.");
    return;
  }
  if (file.size === 0) {
    showError("This file is empty. Choose a valid leaf image.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError("Image file is too large. The maximum supported size is 20 MB.");
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    URL.revokeObjectURL(objectUrl);
    if (image.naturalWidth < MIN_IMAGE_DIMENSION || image.naturalHeight < MIN_IMAGE_DIMENSION) {
      showError(`Image dimensions must be at least ${MIN_IMAGE_DIMENSION} × ${MIN_IMAGE_DIMENSION} pixels.`);
      return;
    }
    if (image.naturalWidth * image.naturalHeight > MAX_IMAGE_PIXELS) {
      showError("Image dimensions are too large to process safely.");
      return;
    }
    selectedFile = file;
    selectedImage = image;
    elements.preview.src = URL.createObjectURL(file);
    elements.previewCard.hidden = false;
    elements.fileName.textContent = file.name;
    elements.classify.disabled = false;
    elements.classifyLabel.textContent = "Classify this leaf";
    elements.resultEmpty.hidden = false;
    elements.resultContent.hidden = true;
    elements.modelMessage.hidden = !predictor;
  };
  image.onerror = () => {
    URL.revokeObjectURL(objectUrl);
    showError("This image could not be opened. It may be corrupted or unsupported.");
  };
  image.src = objectUrl;
}

function removeSelectedFile() {
  resetSelectedFile();
  clearError();
}

function resetSelectedFile() {
  if (elements.preview.src.startsWith("blob:")) URL.revokeObjectURL(elements.preview.src);
  selectedFile = null;
  selectedImage = null;
  elements.input.value = "";
  elements.preview.removeAttribute("src");
  elements.previewCard.hidden = true;
  elements.classify.disabled = true;
  elements.classifyLabel.textContent = "Choose a photo to continue";
}

function displayClasses(classNames) {
  const render = (container) => {
    container.replaceChildren();
    const icon = document.createElement("span");
    icon.className = "supported-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "⌁";
    container.append(icon);
    const content = document.createElement("div");
    if (classNames.length === 0) {
      const message = document.createElement("p");
      message.textContent = "No supported classes are listed yet. They will appear here from the trained model metadata.";
      content.append(message);
    } else {
      classNames.forEach((name) => {
        const chip = document.createElement("span");
        chip.className = "class-chip";
        chip.textContent = name;
        content.append(chip);
      });
    }
    container.append(content);
  };
  render(elements.supported);
  render(elements.modelClasses);
}

function validateMetadata(metadata) {
  if (
    !Array.isArray(metadata.class_names) ||
    metadata.class_names.length < 2 ||
    metadata.class_names.some((name) => typeof name !== "string" || !name) ||
    new Set(metadata.class_names).size !== metadata.class_names.length
  ) {
    throw new Error("Model metadata has an invalid class-name mapping.");
  }
  if (
    JSON.stringify(metadata.image_size) !== JSON.stringify([IMAGE_SIZE, IMAGE_SIZE]) ||
    metadata.preprocessing !== "mobilenet_v2"
  ) {
    throw new Error("Model metadata must declare 224 × 224 images and MobileNetV2 preprocessing.");
  }
}

async function loadModelMetadata() {
  if (modelMetadata) return modelMetadata;
  const metadataResponse = await fetch(METADATA_URL);
  if (!metadataResponse.ok) {
    throw new Error("No trained model metadata was found. Follow the README instructions to add a model.");
  }
  const metadata = await metadataResponse.json();
  validateMetadata(metadata);
  modelMetadata = metadata;
  displayClasses(metadata.class_names);
  return metadata;
}

async function loadPredictor() {
  if (predictor) return predictor;
  if (modelLoadPromise) return modelLoadPromise;
  modelLoadPromise = (async () => {
    if (!window.tf) {
      throw new Error("TensorFlow.js did not load. Check your internet connection and refresh the page.");
    }
    const metadata = await loadModelMetadata();
    const model = await window.tf.loadLayersModel(MODEL_URL);
    const inputShape = model.inputs[0]?.shape;
    const outputShape = model.outputs[0]?.shape;
    if (
      !inputShape ||
      inputShape.length !== 4 ||
      inputShape[1] !== IMAGE_SIZE ||
      inputShape[2] !== IMAGE_SIZE ||
      inputShape[3] !== 3 ||
      !outputShape ||
      outputShape.length !== 2 ||
      outputShape[1] !== metadata.class_names.length
    ) {
      model.dispose();
      throw new Error("The TensorFlow.js model input/output does not match its class metadata.");
    }
    modelMetadata = metadata;
    predictor = model;
    displayClasses(metadata.class_names);
    elements.modelMessage.hidden = true;
    return model;
  })();
  try {
    return await modelLoadPromise;
  } catch (error) {
    modelLoadPromise = null;
    throw error;
  }
}

function preprocess(image) {
  return window.tf.tidy(() => {
    const pixels = window.tf.browser.fromPixels(image, 3);
    const resized = window.tf.image.resizeBilinear(pixels, [IMAGE_SIZE, IMAGE_SIZE]);
    return resized.toFloat().div(127.5).sub(1).expandDims(0);
  });
}

async function runPrediction() {
  if (!selectedImage || !selectedFile) return;
  elements.classify.disabled = true;
  elements.classifyLabel.textContent = "Loading model…";
  clearError();
  try {
    const model = await loadPredictor();
    elements.classifyLabel.textContent = "Analyzing image…";
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    const input = preprocess(selectedImage);
    let output;
    try {
      output = model.predict(input);
      if (Array.isArray(output)) throw new Error("Multiple model outputs are not supported.");
      const scores = await output.data();
      const classNames = modelMetadata.class_names;
      if (scores.length !== classNames.length) {
        throw new Error("The model returned a score count that does not match its class labels.");
      }
      const values = Array.from(scores);
      if (values.some((score) => !Number.isFinite(score) || score < 0 || score > 1)) {
        throw new Error("The model returned invalid class scores.");
      }
      const total = values.reduce((sum, score) => sum + score, 0);
      if (Math.abs(total - 1) > 0.001) {
        throw new Error("The model output is not a valid softmax score vector.");
      }
      const ranked = values
        .map((score, index) => ({ name: classNames[index], score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);
      displayPrediction(ranked);
      addHistory(selectedFile.name, ranked[0]);
    } finally {
      input.dispose();
      if (output && !Array.isArray(output)) output.dispose();
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : "Classification could not be completed.");
    elements.modelMessage.hidden = false;
  } finally {
    elements.classify.disabled = !selectedFile;
    elements.classifyLabel.textContent = "Classify this leaf";
  }
}

function displayPrediction(ranked) {
  const first = ranked[0];
  elements.resultName.textContent = first.name;
  elements.resultScore.textContent = `${(first.score * 100).toFixed(1)}%`;
  elements.topList.replaceChildren();
  ranked.forEach((item) => {
    const row = document.createElement("div");
    row.className = "top-row";
    const name = document.createElement("span");
    name.className = "top-row-label";
    name.textContent = item.name;
    name.title = item.name;
    const score = document.createElement("span");
    score.className = "top-row-score";
    score.textContent = `${(item.score * 100).toFixed(1)}%`;
    const track = document.createElement("span");
    track.className = "score-track";
    const fill = document.createElement("span");
    fill.className = "score-fill";
    fill.style.width = `${Math.max(0, Math.min(100, item.score * 100))}%`;
    track.append(fill);
    row.append(name, score, track);
    elements.topList.append(row);
  });
  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  elements.modelMessage.hidden = true;
}

function addHistory(filename, prediction) {
  predictionHistory.unshift({
    filename,
    className: prediction.name,
    score: `${(prediction.score * 100).toFixed(1)}%`,
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  });
  predictionHistory.splice(10);
  elements.historyBody.replaceChildren();
  predictionHistory.forEach((entry) => {
    const row = document.createElement("tr");
    [entry.filename, entry.className, entry.score, entry.time].forEach((text) => {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    });
    elements.historyBody.append(row);
  });
  elements.historySection.hidden = false;
}

function navigate(page) {
  document.querySelectorAll(".page").forEach((section) => {
    const active = section.id === `${page}-page`;
    section.hidden = !active;
    section.classList.toggle("active", active);
  });
  document.querySelectorAll(".nav-link").forEach((link) => {
    const active = link.dataset.page === page;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

if (window.location.protocol !== "file:") {
  loadModelMetadata().catch((error) => {
    elements.modelMessage.querySelector("p").textContent = error.message;
  });
}

elements.input.addEventListener("change", (event) => setSelectedFile(event.target.files?.[0]));
elements.remove.addEventListener("click", removeSelectedFile);
elements.classify.addEventListener("click", runPrediction);
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.input.click();
  }
});
["dragenter", "dragover"].forEach((type) => {
  elements.dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("drag-over");
  });
});
["dragleave", "drop"].forEach((type) => {
  elements.dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("drag-over");
  });
});
elements.dropZone.addEventListener("drop", (event) => {
  setSelectedFile(event.dataTransfer?.files?.[0]);
});
document.querySelectorAll("[data-page]").forEach((link) => {
  link.addEventListener("click", () => navigate(link.dataset.page));
});
window.addEventListener("hashchange", () => {
  const page = window.location.hash.slice(1);
  navigate(["about", "model"].includes(page) ? page : "home");
});

displayClasses([]);
if (window.location.hash) {
  const page = window.location.hash.slice(1);
  navigate(["about", "model"].includes(page) ? page : "home");
}
