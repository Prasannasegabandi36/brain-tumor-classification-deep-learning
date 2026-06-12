import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = "model/brain_tumor_model.h5"
IMG_SIZE = 150

class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]


def predict_tumor(img_path):
    model = load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    result = class_names[predicted_index]

    return result, confidence


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    result, confidence = predict_tumor(image_path)

    print("Prediction:", result)
    print(f"Confidence: {confidence:.2f}%")
