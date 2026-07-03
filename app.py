import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿"
)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("leaf_model.keras")
model = load_model()
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
st.title(" Plant Leaf Disease Detection")
st.write("Upload a plant leaf image to detect its disease.")
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    st.subheader("Prediction Result")

    st.success(f" Disease: **{class_names[predicted_index]}**")

    st.info(f" Confidence: **{confidence:.2f}%**")

    st.progress(min(int(confidence), 100))
