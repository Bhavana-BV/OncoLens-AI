# OncoLens Website

A Flask-based research showcase for the OncoLens oral-image classification project.

## 1. Setup (Windows / VS Code)

Open this folder in VS Code. In the VS Code terminal:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## 2. Add trained model files

Create a `models` folder and copy the verified model checkpoints into it:
- `mobilenetv2_sharpen.keras`
- `efficientnetb0_sharpen.keras`
- `resnet50_best.keras` (only if you want to include ResNet50)

Do not rename or overwrite your original checkpoints.

## 3. Important inference verification

The interactive prediction endpoint is deliberately disabled initially. Before enabling it, verify:
- model input shape and dtype;
- image resizing and colour-channel order;
- preprocessing/rescaling expected by each model;
- output shape and class-index mapping (0 = non_cancer, 1 = cancer in the experiment metadata);
- the exact selected model checkpoint.

Do not assume that every model uses identical preprocessing. Never tune a decision threshold using the held-out test set.

## 4. Research disclaimer

OncoLens is an experimental academic research prototype. It is not clinically validated and must not be used for diagnosis, screening, treatment decisions, or as a substitute for a qualified healthcare professional.
