# Face Skin Analyzer

This is a Streamlit web application that allows users to upload an image of their face and get predictions about their skin condition based on a pre-trained deep learning model.

## Features

1. **Image Upload**: Users can upload an image of their face in JPG, PNG, or JPEG format.
2. **Skin Condition Prediction**: The app uses a pre-trained FastAI model to predict the skin condition(s) present in the uploaded image. The top 3 predicted conditions are displayed with their respective probabilities.
3. **Recommended Solutions**: The app also provides links to Amazon products that can help treat the predicted skin conditions.

## Dependencies

This app requires the following Python libraries:

- `streamlit`
- `PIL`
- `fastai.vision.all`
- `pandas`
- `plotly.express`
- `numpy`
- `pathlib`

These dependencies can be installed using `pip`:

```
pip install streamlit pillow fastai pandas plotly numpy pathlib
```

## Usage

1. Clone the repository or download the Python script.
2. Ensure that the required dependencies are installed.
3. Place the pre-trained FastAI model file (`export.pkl`) and the Amazon category data (`amazon_categories.xlsx`) in the same directory as the Python script.
4. Run the Streamlit app using the following command:

   ```
   streamlit run face_skin_analyzer.py
   ```

5. The app will open in your default web browser. You can then upload an image of your face and click the "Predict" button to get the skin condition predictions and recommended solutions.

## Troubleshooting

If you encounter any issues while running the app, such as an error loading the pre-trained model, the app will display an error message with the relevant information. You can try the following steps to troubleshoot:

1. Ensure that the `export.pkl` file is present in the same directory as the Python script.
2. Check the error message for any additional information that may help identify the issue.
3. If you're still having trouble, you can reach out to the app's maintainers for assistance.

## License

This project is licensed under the [MIT License](LICENSE).
