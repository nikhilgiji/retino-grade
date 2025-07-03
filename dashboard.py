import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import pandas as pd

# Page config for wide layout and initial sidebar expanded
st.set_page_config(
    page_title="RetinoGrade Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional CSS to reduce sidebar width and control image height and padding
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 220px;
    }
    img {
        max-height: 400px;
        object-fit: contain;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .css-1d391kg {
        padding: 0.5rem 1rem 0.5rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data  # Updated caching command as per new Streamlit version
def load_model():
    return tf.keras.models.load_model('models/best_model_cbam.h5', compile=False)

model = load_model()

CLASS_NAMES = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']

def preprocess_image(image: Image.Image, target_size=(224,224)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def get_gradcam_heatmap(model, image, last_conv_layer_name="conv5_block3_out", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    return heatmap

def overlay_heatmap(img: np.array, heatmap: np.array, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    overlay = heatmap_colored * alpha + img
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

st.title("RetinoGrade DR Grading - Advanced Inference Dashboard")

st.sidebar.header("Settings")

thresholds = {}
for i, cls_name in enumerate(CLASS_NAMES):
    thresholds[i] = st.sidebar.slider(f"Threshold for {cls_name}", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("""
---
**Class Descriptions:**

- 0: No Diabetic Retinopathy  
- 1: Mild  
- 2: Moderate  
- 3: Severe  
- 4: Proliferative Diabetic Retinopathy  
""")

uploaded_files = st.file_uploader("Upload fundus images (PNG/JPG)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    results = []

    for file in uploaded_files:
        image = Image.open(file).convert('RGB')
        input_tensor = preprocess_image(image)
        preds = model.predict(input_tensor)[0]

        pred_binary = (preds >= np.array(list(thresholds.values()))).astype(int)

        if pred_binary.sum() == 0:
            pred_label_idx = 0
        elif pred_binary.sum() == 1:
            pred_label_idx = np.argmax(pred_binary)
        else:
            candidates = np.where(pred_binary == 1)[0]
            best = candidates[np.argmax(preds[candidates])]
            pred_label_idx = best

        pred_label = CLASS_NAMES[pred_label_idx]

        heatmap = get_gradcam_heatmap(model, input_tensor, last_conv_layer_name="conv5_block3_out", pred_index=pred_label_idx)
        img_array = np.array(image.resize((224, 224)))
        heatmap_img = overlay_heatmap(img_array, heatmap)

        st.subheader(f"Image: {file.name}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width=350)
        with col2:
            st.image(heatmap_img, caption="Grad-CAM Overlay", width=350)

        st.write(f"Predicted Class: **{pred_label}**")

        st.write("Confidence Scores:")
        cols = st.columns(len(CLASS_NAMES))
        for idx, cls_name in enumerate(CLASS_NAMES):
            with cols[idx]:
                st.write(f"**{cls_name}**")
                st.write(f"{preds[idx]*100:.2f}%")

        results.append({
            "filename": file.name,
            "predicted_label": pred_label,
            **{cls_name: float(preds[idx]) for idx, cls_name in enumerate(CLASS_NAMES)}
        })

    df_results = pd.DataFrame(results)
    st.subheader("Batch Prediction Results")
    st.dataframe(df_results)

    csv_buffer = io.StringIO()
    df_results.to_csv(csv_buffer, index=False)
    st.download_button("Download Prediction Results CSV", csv_buffer.getvalue(), "retinograde_predictions.csv", "text/csv")

else:
    st.info("Please upload one or more fundus images to start inference.")