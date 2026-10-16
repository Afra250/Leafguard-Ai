# LeafGuard — AI-Based Crop Disease Detection

LeafGuard is a beginner-friendly **HTML, CSS, and JavaScript** browser
application for ranking crop-leaf image classes. The application runs in the
browser, validates and previews JPG/PNG uploads, and can make real predictions
when a compatible TensorFlow.js model is supplied.

No dataset or trained model is included, and no accuracy or evaluation result
is claimed. Without a model, the interface explains what is missing and does
not make up predictions or supported classes.

> **Important:** LeafGuard is an educational decision-support prototype.
> Predictions may be incorrect and are limited to classes in the trained model.
> Model scores do not guarantee diagnostic correctness. Do not use predictions
> alone to choose pesticides or treatments. Consult a qualified agricultural
> expert for important farming decisions.

## Problem and objectives

Crop symptoms can be difficult to recognize from a photograph. LeafGuard
demonstrates a transparent image-classification workflow: accept one leaf
image, show the highest-scoring supported label and up to three model scores,
and explain that the output is not a diagnosis.

The project aims to:

- Provide a responsive browser interface for leaf image upload and preview.
- Validate file format, image dimensions, and corrupt files.
- Use MobileNetV2-compatible preprocessing and a saved class mapping.
- Display only labels recorded by actual model metadata.
- Keep uploaded image data and prediction history in the current page session.
- Document dataset, evaluation, privacy, and responsible-use limitations.

## Features

- Responsive Home, About Project, and Model Information pages.
- Drag-and-drop or browse for JPG, JPEG, or PNG images.
- File-size, pixel-dimension, image-decoding, and MIME/extension checks.
- Image preview, model loading status, and clear errors.
- Top prediction and up to three ranked class scores from a supplied model.
- Supported-class list populated from trained model metadata.
- Session-only history with filename, predicted class, score, and time.
- Images are processed in the browser and not saved by this application.
- Optional MobileNetV2 training and test-evaluation scripts.

## Technology

### Web application

- HTML5
- CSS3 (responsive layout)
- Vanilla JavaScript (no frontend framework/build step)
- TensorFlow.js 4.22 for browser inference (loaded from jsDelivr)

### Optional model preparation

The included transfer-learning trainer and evaluator use Python, TensorFlow /
Keras, Pillow, NumPy, scikit-learn, and Matplotlib. These are optional
development tools: the user-facing web application itself is JavaScript,
HTML, and CSS. A trained Keras model must be converted to TensorFlow.js format
before browser inference.

## Project structure

```text
LeafGuard/
├── index.html                   # Browser app and page content
├── styles.css                   # Responsive green-and-white design
├── app.js                       # Uploads, validation, inference, history, nav
├── train.py                     # Optional MobileNetV2 training command
├── evaluate.py                  # Optional separate test-set evaluation
├── requirements.txt             # Optional Python training/test dependencies
├── src/
│   ├── preprocessing.py         # Python training/inference preprocessing helper
│   ├── prediction.py            # Python Keras model helper
│   ├── training.py              # Transfer learning and held-out evaluation
│   └── metrics.py               # Metrics and confusion-matrix plot
├── data/README.md               # Dataset source and split guidance
├── data/                        # Add authorized train/validation/test images
├── models/                      # Local model artifacts (weights are gitignored)
│   └── README.md                # Model conversion and expected files
├── models/tfjs/                 # Converted browser model.json and weight shards
├── reports/                     # Generated metrics and plots
├── tests/                       # Tests for optional Python model helpers
└── docs/
    ├── project_report.md
    ├── concept_note.md
    ├── lean_canvas.md
    └── presentation_guide.md
```

## Run the web application (Windows)

The browser needs a local HTTP server to load model and metadata files. You do
not need Node.js to run the basic site:

1. Install Visual Studio Code and its **Live Server** extension, or use another
   static-file HTTP server.
2. Open the `LeafGuard` folder in VS Code.
3. Right-click `index.html` and choose **Open with Live Server**.
4. The application opens in your browser. Try uploading a JPG or PNG; real
   classification remains unavailable until a model is supplied.

Opening the file directly with `file://` may show the layout, but browser
security blocks `fetch()` model/metadata loading. Use a local HTTP server.

TensorFlow.js is loaded from jsDelivr, so browser inference also needs internet
access unless you download and serve the library locally. The app provides an
explicit error if TensorFlow.js cannot load.

## Dataset preparation

No image dataset is included. PlantVillage is a research dataset to investigate;
see [`data/README.md`](data/README.md) and the
[authors' repository](https://github.com/spMohanty/PlantVillage-Dataset).
Before using an archive, check the current terms for that exact copy and
source, ensure your intended use is allowed, and preserve attribution.
Third-party copies may have different terms. Current terms could not be
independently verified while preparing this project, so this is a source lead,
not a claim that every copy is openly licensed.

Arrange authorized images by their real class names:

```text
data/
├── train/
│   ├── Actual_Class_One/
│   └── Actual_Class_Two/
├── validation/
│   ├── Actual_Class_One/
│   └── Actual_Class_Two/
└── test/
    ├── Actual_Class_One/
    └── Actual_Class_Two/
```

These folder names are placeholders, not supported LeafGuard labels. Keep the
same class folders in every split. Split by original plant/capture or
near-duplicate group to reduce leakage, and do not use the test set to tune
training. PlantVillage images often have controlled backgrounds; performance
on them does not establish performance on real farm photographs.

## Optional model training

Install Python 3.10/3.11 and dependencies if you want to train your own model:

```powershell
cd "$HOME\Desktop\LeafGuard"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --data-dir data --model-dir models --reports-dir reports --epochs 20
```

The trainer reads labels from `data/train`, checks validation/test class
ordering, uses an ImageNet-pretrained MobileNetV2 feature extractor and a
classification head, and selects a checkpoint by validation loss. It evaluates
the selected model on the separate test set. It writes a Keras model, ordered
label metadata, measured test metrics, a classification report, and a
confusion-matrix image. Training has not been run for this project instance;
there are no measured results yet.

## Convert the trained model for browser inference

The browser requires a TensorFlow.js **LayersModel**, not the native `.keras`
file. After successful training, install the TensorFlow.js converter in the
Python environment and convert the trained model:

```powershell
python -m pip install tensorflowjs
tensorflowjs_converter --input_format=keras models\leafguard_model.keras models\tfjs
```

The output directory must contain `models/tfjs/model.json` and its weight
shards. Keep `models/model_metadata.json` created by training beside the
`tfjs` directory. The browser reads `class_names`, `image_size`, and
`preprocessing` from that metadata and checks model input/output dimensions.

If the converter reports incompatibility with the installed TensorFlow/Keras
version, use a converter release compatible with that model and record the
versions used. Do not modify the labels or preprocessing declaration to bypass
the browser's checks; correct compatibility issues at conversion/export time.
Large model files are ignored by Git and should not be committed.

## Test and evaluate

Optional Python helper tests:

```powershell
python -m pytest -q
```

Re-evaluate an existing model on the untouched test directory:

```powershell
python evaluate.py --data-dir data --model-dir models --reports-dir reports
```

The web UI itself can be checked by opening it in Live Server, uploading a
valid image, trying an invalid file, inspecting responsive layouts, and
checking the browser console/network tab. Browser scores come from the
installed model and are not diagnostic certainty.

## Privacy

The image is decoded and processed in browser memory. The app does not upload
or save image bytes. Prediction history stores only filename, class, score,
and time in a JavaScript in-memory array for the current page session. It
clears on refresh. Avoid sensitive photos. If you host the static site, review
the hosting provider's access and request-log policies; a remote model host or
analytics integration would change the privacy behavior.

## Troubleshooting

- **Model not installed:** train a model, convert it to TensorFlow.js, and
  place `model.json`, its weight shards, and generated metadata in the paths
  described above.
- **Page displays but cannot load metadata:** serve the project over localhost
  HTTP (for example, VS Code Live Server), not `file://`.
- **TensorFlow.js did not load:** check internet access to jsDelivr or host the
  TensorFlow.js library locally.
- **Metadata/model mismatch:** verify the model is a converted MobileNetV2
  classifier with 224 × 224 × 3 inputs, output count matching its class list,
  and the declared MobileNetV2 preprocessing.
- **Invalid image:** use a readable JPG/PNG under 20 MB, at least 32 × 32
  pixels, and no more than 25 million pixels.
- **Poor field performance:** curated backgrounds may not represent farm
  conditions; evaluate on independent, representative, expert-labeled field
  photographs.

## Limitations and future work

This classifier covers only the labels learned from the supplied dataset.
Unseen symptoms may be forced into a known class. Scores may be
overconfident and are not guaranteed calibrated probabilities. Controlled
images may not represent cultivars, lighting, cameras, regions, or field
conditions. The prototype has no agronomic validation and provides no
treatment recommendations.

Future work could include expert-reviewed field datasets, independent
real-world testing, score calibration, out-of-distribution warnings,
accessibility/localization, a reproducibility manifest, and expert feedback
workflows.

## Project report and presentation

See [`docs/project_report.md`](docs/project_report.md) for the major-project
report outline, [`docs/lean_canvas.md`](docs/lean_canvas.md) for assumptions
that need validation, and [`docs/presentation_guide.md`](docs/presentation_guide.md)
for a PowerPoint outline. Include actual measured results only; if training has
not run, state “Not evaluated yet.”
