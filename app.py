# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import tensorflow as tf

st.set_page_config(page_title="Cats vs Dogs Classifier", layout="centered")

st.title("Cats vs Dogs — Image Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

MODEL_PATH = "CatsVsDogsModel.keras"

@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Failed to load model at `{path}`: {e}")
        raise
    return model

def preprocess_image(image: Image.Image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((256, 256)) 

    arr = np.array(image).astype("float32") / 255.0

    # Ensure shape (H, W, 3)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] != 3:
        arr = arr[..., :3]

    arr = np.expand_dims(arr, axis=0)  # (1, 256, 256, 3)

    return arr

def predict_image(model, image_arr):
    preds = model.predict(image_arr)

    # Normalize shape
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    # Case: binary sigmoid → output shape (1,1)
    if preds.shape[-1] == 1:
        prob_dog = float(preds[0, 0])
        prob_cat = 1 - prob_dog
        confidences = np.array([prob_cat, prob_dog])
    else:
        # Softmax output
        confidences = tf.nn.softmax(preds[0]).numpy()

    label_index = int(np.argmax(confidences))
    return label_index, confidences


# Load model
with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    st.image(image, caption="Input Image", use_container_width=True)


    try:
        image_arr = preprocess_image(image)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    with st.spinner("Predicting..."):
        try:
            label_idx, confidences = predict_image(model, image_arr)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    labels = ["Cat", "Dog"]
    predicted_label = labels[label_idx]

    st.markdown("### Prediction")
    st.write(predicted_label)

    st.markdown("### Confidence Scores")
    st.write(f"- Cat: {confidences[0]*100:.2f}%")
    st.write(f"- Dog: {confidences[1]*100:.2f}%")

else:
    st.info("Upload an image to classify.")

st.markdown("---")
st.write("Modify the `labels` list if your model uses a different label order.")
