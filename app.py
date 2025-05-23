import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# loading the model
model = load_model('smoker_model.h5')

# preprocessing function
def preprocess_image(img, target_size=(300, 300)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img):
    processed = preprocess_image(img)
    pred = model.predict(processed)[0][0]
    return "Smoking" if pred > 0.5 else "Not Smoking", float(pred)


st.set_page_config(page_title="Smoking Detector", layout="centered")
st.title("🚬 Smoking Detection")
st.write("Upload an image or use your camera to detect if someone is smoking.")

# image Upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# camera input
camera_image = st.camera_input("Take a picture")

# prediction and display
if uploaded_file or camera_image:
    st.subheader("Result:")
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = Image.open(camera_image)

    st.image(image, caption="Input Image", use_column_width=True)
    label, confidence = predict(image)
    st.success(f"Prediction: **{label}** ({confidence:.2f} confidence)")

