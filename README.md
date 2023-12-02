# Arabic Sarcasm Detection WebApp
This repository contains the source code for a Streamlit web application that leverages both traditional machine learning models and the AraBERT transformer model to detect sarcasm in Arabic text. It includes a TensorFlow implementation for AraBERT and uses several classical machine learning models for comparative analysis.

## Features
Sarcasm detection in Arabic text using AraBERT and multiple machine learning models.
Visualization of confidence levels using Plotly bar and gauge charts.
Use of Streamlit for an interactive web application interface.
##Installation
Before running the application, ensure you have the following prerequisites installed:

- Python 3.8+
- Numpy
- Pandas
- Streamlit
- TensorFlow 2.x
- Transformers library
- Plotly
- AraBERT

You can install the necessary libraries using pip:
```
pip install streamlit tensorflow transformers plotly arabert numpy pandas
```
## Usage
To run the Streamlit application, navigate to the repository's root directory and execute:

```
streamlit run app.py
```
## Application Structure
- **SarcasmViz.py:** The main Streamlit application script.
- **SarcasmAraBERT:** The directory containing the trained TensorFlow model (AraBERT).
- ***.pkl:** Pickle files for the trained classical machine learning models.
## Functionality
- **Model Loading:** The TensorFlow model and Pickle models are loaded using Streamlit's caching to improve performance.
- **Text Input:** Users can input Arabic text into the application, which is then processed and analyzed by both the AraBERT model and classical ML models.

## Sarcasm Prediction and Visualization:

Each model predicts whether the text is sarcastic or not, along with a confidence score.
The results are visualized using Plotly bar charts for comparative analysis.

**Display Results:** The application displays the prediction results in a tabular format for easy comparison.
