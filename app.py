import os
import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/brain_tumor_model.h5"
IMG_SIZE = 150

# Important: This order should match your training class order.
class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. First train the model using: python train_model.py"
        )
        st.stop()
    return load_model(MODEL_PATH)


def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array


def main():
    st.set_page_config(
        page_title="Brain Tumor MRI Classification",
        page_icon="🧠",
        layout="centered"
    )

    st.title("🧠 Brain Tumor MRI Classification")
    st.write(
        "Upload a brain MRI image and this deep learning model will classify it "
        "as Glioma Tumor, Meningioma Tumor, Pituitary Tumor, or No Tumor."
    )

    st.warning(
        "Educational project only. This app is not a real medical diagnosis tool."
    )

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        model = load_trained_model()
        img, processed_img = preprocess_image(uploaded_file)

        st.image(img, caption="Uploaded MRI Image", use_container_width=True)

        if st.button("Predict"):
            prediction = model.predict(processed_img)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            result = class_names[predicted_index]

            st.subheader("Prediction Result")
            st.success(f"Predicted Class: {result}")
            st.info(f"Confidence Score: {confidence:.2f}%")

            st.subheader("Class Probabilities")
            for i, class_name in enumerate(class_names):
                probability = float(prediction[0][i])
                st.write(f"{class_name}: {probability * 100:.2f}%")
                st.progress(probability)


if __name__ == "__main__":
    main()
