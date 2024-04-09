import streamlit as st
from PIL import Image as PILImage
from fastai.vision.all import *
import pandas as pd
from PIL import Image
import plotly.express as px
import numpy as np
import pathlib
import urllib.parse

plt = platform.system()
if plt == "Linux":
    pathlib.WindowsPath = pathlib.PosixPath
try:
    learn = load_learner("export.pkl")
    labels = learn.dls.vocab
except Exception as e:
    st.error(f"Error loading model: {e}")
    # Optionally, provide instructions for troubleshooting


def predict(image):
    """Predicts the face condition based on the uploaded image"""
    # Convert PIL image to tensor
    img = PILImage.create(image)

    # Make prediction
    pred, pred_idx, probs = learn.predict(img)

    # Return prediction dictionary
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


st.title("Face Skin AI")
st.markdown(
    """This app, called "Face Skin AI," helps you detect potential skin issues on your face. Simply upload a photo of your face, and it will analyze the image to provide you with possible face skin issues you might have. After uploading, click on "Predict" to get the results. It then suggests solutions for those conditions, directing you to relevant products on Amazon for treatment."""
)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Basic image format check
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        try:
            predictions = predict(image)
            st.subheader("Predicted Conditions:")
            preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            top_class = 3
            preds = preds[:top_class]
            preds2 = [x[0] for x in preds]
            # Read recommendations from excel (assuming it's in the same directory)
            df2 = pd.read_excel("amazon_categories.xlsx")
            for pred, probs in preds:
                st.write(f"{pred}: {np.round(probs*100,2)}%")
            preds_graph = dict((x, np.round(y, 6)) for x, y in preds)
            df = pd.DataFrame(
                {
                    "Skin Problem": preds_graph.keys(),
                    "Probabilities": preds_graph.values(),
                }
            )
            fig = px.bar(
                df,
                x="Probabilities",
                y="Skin Problem",
                orientation="h",
                text="Probabilities",
            )
            st.plotly_chart(fig)
            st.subheader("Recommended Solutions:")
            st.write(f"Find solution to your problems on amazon via below links")
            for idx, row in df2.iterrows():
                if row["class"] in preds2:
                    condition = row["class"]
                    encoded_condition = condition.replace(" ", "_")
                    print(encoded_condition)
                    link = row["profit_link"]
                    st.write(f"- [{condition} treatment]( {link} )")
                    st.markdown(
                        f"""<a href={link}><img src="./app/static/{encoded_condition}.png" width="100%" height="auto" alt="{condition} Treatment"></a>""",
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Error making prediction: {e}")
