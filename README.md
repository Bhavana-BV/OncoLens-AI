# OncoLens: Leakage-Controlled, Uncertainty-Aware Oral Cancer Detection

An academic deep learning research project investigating oral cancer image classification using three convolutional neural network architectures: **MobileNetV2, EfficientNetB0, and ResNet50**.

OncoLens combines image preprocessing, leakage-controlled dataset splitting, comparative model evaluation, uncertainty estimation, and explainability using Gradient-weighted Class Activation Mapping (Grad-CAM).

> **Research Disclaimer:** OncoLens is an experimental academic research prototype. It has not been clinically validated and must not be used for medical diagnosis, treatment decisions, or clinical screening.

## Project Overview

Oral cancer detection through medical image analysis is an important area of research in computer vision and healthcare AI.

This project explores how different CNN architectures perform on an oral image classification task while incorporating measures to reduce data leakage and improve model interpretability.

The project focuses on:

* Leakage-controlled dataset preparation.
* Image preprocessing and enhancement.
* Comparative evaluation of three CNN architectures.
* Performance assessment using multiple evaluation metrics.
* Model explainability using Grad-CAM.
* Uncertainty-aware prediction analysis.
* A Flask-based web demonstration.

## Deep Learning Models

Three pretrained CNN architectures were investigated:

| Model          | Architecture                                 |
| -------------- | -------------------------------------------- |
| MobileNetV2    | Lightweight convolutional neural network     |
| EfficientNetB0 | Compound-scaled convolutional neural network |
| ResNet50       | Deep residual convolutional neural network   |

The trained model files are available in the `models/` directory.

## Model Evaluation Results

Evaluation was performed on a held-out test set containing **325 images**.

| Evaluation Metric | MobileNetV2 | EfficientNetB0 | ResNet50 |
| ----------------- | ----------: | -------------: | -------: |
| Accuracy          |      90.46% |         87.69% |   91.08% |
| Precision         |      90.45% |         87.71% |   88.02% |
| Recall            |      92.00% |         89.71% |   96.57% |
| Specificity       |      88.67% |         85.33% |   84.67% |
| F1-score          |      91.22% |         88.70% |   92.10% |
| ROC-AUC           |      95.96% |         96.65% |   97.07% |
| PR-AUC            |      96.29% |         97.07% |   97.32% |

**Observation:** ResNet50 achieved the highest test accuracy, recall, F1-score, ROC-AUC, and PR-AUC among the three evaluated models. These results represent performance on this experimental dataset and do not establish clinical effectiveness.

## Key Features

* Three-model deep learning comparison.
* Oral image classification.
* Image preprocessing and enhancement.
* Leakage-controlled dataset splitting.
* Grad-CAM visual explanations.
* Multiple classification evaluation metrics.
* Research notebooks and experimental results.
* Flask-based web demonstration.

## Repository Structure
OncoLens-AI/
│
├── models/
│   ├── mobilenetv2_sharpen.keras
│   ├── efficientnetb0_sharpen.keras
│   └── resnet50_best.keras
│
├── website/
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/
│   └── static/
│
├── streamlit_app/
│
├── research/
│   ├── documentation/
│   ├── figures/
│   ├── notebooks/
│   └── results/
│
├── .gitignore
└── README.md

## Technologies Used

| Category                | Technologies              |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Deep Learning           | TensorFlow, Keras         |
| Image Processing        | OpenCV                    |
| Data Analysis           | NumPy, Pandas             |
| Machine Learning        | Scikit-learn              |
| Visualization           | Matplotlib                |
| Web Development         | Flask                     |
| Development Environment | Jupyter Notebook, VS Code |


## Running the Flask Web Application

### 1. Clone the repository

```bash
git clone https://github.com/Bhavana-BV/OncoLens-AI.git
```

### 2. Navigate to the project directory

```bash
cd OncoLens-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r website/requirements.txt
pip install tensorflow opencv-python numpy
```

### 6. Run the application

```bash
python website/app.py
```

Open the local URL displayed in your terminal to access the application.

**Note:** Model inference requires the trained model files in the `models/` directory and compatible Python dependencies.

## Research Limitations

* The dataset used in this project may not represent the full diversity of real-world clinical images.
* Model predictions may be incorrect.
* Grad-CAM visualizations are explanatory aids and do not establish clinical reasoning.
* Uncertainty estimates do not guarantee prediction reliability.
* The models have not undergone clinical validation or prospective testing.

**OncoLens is strictly an academic research prototype and is not a substitute for professional medical evaluation.**

## Future Enhancements

* Improve model generalization using larger and more diverse datasets.
* Explore model calibration and uncertainty estimation.
* Develop an interactive web-based research demonstration.
* Investigate lightweight deployment for mobile platforms.
* Conduct further external validation.

## Author

Developed as an academic Artificial Intelligence and Machine Learning research project.

**GitHub:** [Bhavana-BV](https://github.com/Bhavana-BV)

**Repository:** [OncoLens-AI](https://github.com/Bhavana-BV/OncoLens-AI)
