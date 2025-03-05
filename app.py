import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Title and subheading
st.title("Thyroid Cancer Predictor")
st.subheader("Enter Details Manually or Upload an Image")

# Sidebar for details entry
st.sidebar.header("Enter Details Manually")

# Age input
age = st.sidebar.text_input("Age", placeholder="Enter age")

# Sex dropdown
sex = st.sidebar.selectbox("Sex", options=["M", "F"])

# Composition dropdown
composition = st.sidebar.selectbox(
    "Composition",
    options=[
        "solid",
        "predominantly solid",
        "spongiform",
        "predominantly cystic",
        "dense",
        "cystic"
    ]
)

# Echogenicity dropdown
echogenicity = st.sidebar.selectbox(
    "Echogenicity",
    options=[
        "isoechogenicity ",
        "hypoechogenicity",
        "hyperechogenicity",
        "marked hypoechogenicity"
    ]
)

# Margins dropdown
margins = st.sidebar.selectbox(
    "Margins",
    options=[
        "well defined",
        "well defined smooth",
        "ill defined",
        "microlobulated",
        "spiculated",
        "macrolobulated"
    ]
)

# Calcifications dropdown
calcifications = st.sidebar.selectbox(
    "Calcifications",
    options=[
        "microcalcifications",
        "non",
        "microcalcification",
        "macrocalcifications",
        "macrocalcification"
    ]
)

# TIRADS dropdown
tirads = st.sidebar.selectbox(
    "TIRADS",
    options=[
        "4a",
        "4b",
        "4c",
        "2",
        "5",
        "3"
    ]
)

# Malignant percentage input
malignant_percentage = st.sidebar.text_input(
    "Malignant Percentage",
    placeholder="Enter percentage"
)
manual_model = pickle.load(open("optimized_decision_tree_model.pkl", "rb"))
# Button for prediction with manual entry
if st.sidebar.button("Predict Cancer Risk (Manual Entry)"):
        
        features = np.array([
            int(age),
            sex,
            composition,
            echogenicity,
            margins,
            calcifications,
            tirads,
            float(malignant_percentage)
        ]).reshape(1, -1)
        
        # Predict cancer risk percentage
        risk = manual_model.predict(features)[0]
        st.write(f"**Predicted Cancer Risk (Manual Entry):** {risk:.2f}%")

# Image upload and prediction
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Cancer Risk (Image)"):
        try:
            # Load the trained model for image-based predictions
            image_model = load_model("stage_classifier_cnn.h5")

            # Preprocess the uploaded image
            image = Image.open(uploaded_file)

            # Predict cancer risk percentage
            image_risk = image_model.predict(image)
            st.write(f"**Predicted Cancer Risk (Image):** {image_risk:.2f}%")
        except Exception as e:
            st.write("Error during image prediction. Ensure valid inputs and model file.")
else:
    st.write("No image uploaded.")

