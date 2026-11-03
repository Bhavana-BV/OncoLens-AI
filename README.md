# OncoLens: Leakage-Controlled, Uncertainty-Aware Oral Cancer Detection



OncoLens is an academic research project that explores deep learning-based oral cancer image classification using three convolutional neural network architectures: MobileNetV2, EfficientNetB0, and ResNet50.



The project combines image preprocessing, leakage-controlled dataset splitting, model evaluation, explainability using Grad-CAM, and a web-based demonstration.



> **Research disclaimer:** OncoLens is an experimental academic research prototype. It has not been clinically validated and must not be used for medical diagnosis or treatment decisions.



## Models



* MobileNetV2

* EfficientNetB0

* ResNet50



## Repository Structure

OncoLens_GitHub/
|-- models/
|-- website/
|-- streamlit_app/
|-- research/
|   |-- documentation/
|   |-- figures/
|   |-- notebooks/
|   |-- results/
|-- .gitignore
|-- README.md## Evaluation Results



Evaluation on the held-out test set of 325 images:



| Metric      | MobileNetV2 | EfficientNetB0 | ResNet50 |

| ----------- | ----------: | -------------: | -------: |

| Accuracy    |      90.46% |         87.69% |   91.08% |

| Precision   |      90.45% |         87.71% |   88.02% |

| Recall      |      92.00% |         89.71% |   96.57% |

| Specificity |      88.67% |         85.33% |   84.67% |

| F1-score    |      91.22% |         88.70% |   92.10% |

| ROC-AUC     |      95.96% |         96.65% |   97.07% |

| PR-AUC      |      96.29% |         97.07% |   97.32% |



These are research evaluation results, not evidence of clinical performance.



## Features



* Three-model comparison

* Oral image classification

* Image preprocessing

* Grad-CAM visual explanations

* Evaluation using multiple performance metrics

* Research notebooks and figures

* Flask-based web demonstration



## Technologies



* Python

* TensorFlow / Keras

* OpenCV

* NumPy

* Pandas

* Scikit-learn

* Flask

* Streamlit

* Matplotlib

* Jupyter Notebook



## Running the Flask Website



Install the required packages:



```bash

pip install -r website/requirements.txt

pip install tensorflow opencv-python numpy

```



Run:



```bash

python website/app.py

```



Open the local address displayed in the terminal.



## Research Limitations



* This project is intended for academic and experimental purposes.

* Model predictions may be incorrect.

* Grad-CAM visualizations are explanatory aids, not proof of clinical reasoning.

* The models have not been clinically validated.

* Do not use this application to diagnose oral cancer.



## Author



Developed as an academic Artificial Intelligence and Machine Learning research project.



