# OncoLens

## Project Title

OncoLens: Leakage-Controlled, Uncertainty-Aware Oral Cancer Detection:
MobileNetV2 vs EfficientNetB0

## Project Overview

OncoLens is a research project comparing MobileNetV2 and EfficientNetB0
for binary classification of oral images into cancer and non-cancer
categories.

The research workflow includes:

- Dataset auditing and duplicate-aware group splitting
- Image preprocessing comparisons
- Group-aware five-fold cross-validation
- Held-out test evaluation
- Grad-CAM explainability
- Monte Carlo Dropout uncertainty analysis
- TensorFlow Lite Float16 conversion
- Model size and inference latency benchmarking
- Keras-to-TFLite conversion verification

## Folder Structure

- source_code/ : Python scripts and Jupyter notebooks copied from
  the project root.
- results/ : Consolidated experiment tables, report, and metadata.
- figures/ : Selected research figures.
- models/ : Optional trained model files, if included.

## Models

- MobileNetV2
- EfficientNetB0

## Important Evaluation Note

Reported metrics describe experiments on the current dataset.
They do not establish clinical validity or suitability for diagnosis.
External validation and clinical evaluation would be required before
any real-world medical use.

## Reproducibility

The original project directory and dataset are not modified by the
submission-preparation script.

Review the copied source files, experiment settings, dependency
versions, and results before submitting or publishing the project.
