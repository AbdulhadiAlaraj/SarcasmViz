# Arabic Sarcasm Detection WebApp
This repository contains the source code for a Streamlit web application that leverages traditional machine learning models to detect sarcasm in Arabic text.

## Features
Sarcasm detection in Arabic text using multiple machine learning models.
Visualization of confidence levels using Plotly bar charts.
Use of Streamlit for an interactive web application interface.
## Installation
Before running the application, ensure you have the following prerequisites installed:

- Python 3.8+
- Numpy
- Pandas
- Streamlit
- TensorFlow 2.x
- Transformers library
- Plotly
- 
You can install the necessary libraries using pip:
```
pip install streamlit tensorflow transformers plotly numpy pandas
```
## Usage
To run the Streamlit application, navigate to the repository's root directory and execute:

```
streamlit run app.py
```
## Application Structure
- **SarcasmViz.py:** The main Streamlit application script.
- ***.pkl:** Pickle files for the trained classical machine learning models.
## Functionality
- **Model Loading:** The Pickle models are loaded using Streamlit's caching to improve performance.
- **Text Input:** Users can input Arabic text into the application, which is then processed and analyzed by the classical ML models.

## Sarcasm Prediction and Visualization:

Each model predicts whether the text is sarcastic or not, along with a confidence score.
The results are visualized using Plotly bar charts for comparative analysis.

**Display Results:** The application displays the prediction results in a tabular format for easy comparison.
