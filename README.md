# Brain Tumor Detection using CNN and Streamlit

This is a deep learning portfolio project that classifies brain MRI images into four categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The project uses a Convolutional Neural Network built with TensorFlow/Keras and a Streamlit web app for image upload and prediction.

## Technologies Used

- Python
- TensorFlow / Keras
- CNN
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

## Dataset

Use the Brain Tumor MRI Dataset from Kaggle.

Expected folder structure:

```text
brain-tumor-detection-cnn-streamlit/
│
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   │
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── model/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add dataset

Download the dataset and place it inside the `dataset` folder.

Your folders should look like this:

```text
dataset/Training/glioma
dataset/Training/meningioma
dataset/Training/notumor
dataset/Training/pituitary

dataset/Testing/glioma
dataset/Testing/meningioma
dataset/Testing/notumor
dataset/Testing/pituitary
```

### 3. Train the model

```bash
python train_model.py
```

After training, the model will be saved here:

```text
model/brain_tumor_model.h5
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Features

- MRI image upload
- CNN-based prediction
- Confidence score
- Class probability display
- Accuracy and loss graphs after training
- Clean Streamlit web interface

## Resume Points

**Brain Tumor MRI Classification using Deep Learning — 2026**  
Tech Stack: Python, TensorFlow, CNN, Streamlit, NumPy, Matplotlib

- Developed a CNN-based deep learning model to classify brain MRI images into glioma, meningioma, pituitary tumor, and no tumor categories.
- Applied image preprocessing, normalization, data augmentation, and train-validation testing for improved model performance.
- Evaluated model using accuracy, loss curves, classification report, and confusion matrix.
- Built a Streamlit web app with MRI image upload, prediction result, confidence score, and class probability visualization.

## Disclaimer

This project is only for educational and portfolio purposes. It should not be used for real medical diagnosis.
