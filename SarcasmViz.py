import streamlit as st
import plotly.express as px
from arabert import ArabertPreprocessor
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
import pickle
import pandas as pd

@st.cache_resource()
def load_tf_model():
    model_path = r'SarcasmAraBERT'  # Update this path to your model
    return tf.keras.models.load_model(model_path)

@st.cache_resource()
def load_pickle_model(model_name):
    with open(model_name, 'rb') as file:
        return pickle.load(file)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02')
tf_model = load_tf_model()

preprocess = ArabertPreprocessor(model_name='aubmindlab/bert-base-arabertv02')

def predict_with_tf_model(text):
    inputs = tokenize_and_preprocess_single(text, tokenizer, preprocess)

    input_ids = tf.convert_to_tensor(inputs['input_ids'], dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(inputs['attention_mask'], dtype=tf.int32)
    token_type_ids = tf.convert_to_tensor(inputs['token_type_ids'], dtype=tf.int32)

    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }

    predictions = tf_model(model_inputs)
    logits = predictions['logits']
    probabilities = tf.nn.softmax(logits, axis=-1)
    return probabilities.numpy()[0]

def tokenize_and_preprocess_single(text, tokenizer, preprocess):
    preprocessed_text = preprocess.preprocess(text)
    encoded = tokenizer(preprocessed_text, padding=True, truncation=True, max_length=128, return_tensors="tf")
    return encoded

def extract_features(texts, model_name="aubmindlab/bert-base-arabert"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModel.from_pretrained(model_name)

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
        outputs = model(inputs)
        # Extract the embeddings from the output
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
def predict_with_ml_model(model, text, feature_extraction_func):
    # Extract features for the input text
        features = feature_extraction_func([text])  # Text is wrapped in a list
        prediction = model.predict(features)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
        else:
            probabilities = get_confidence_from_decision_function(model, features)
        return prediction, probabilities  # or return probabilities

def get_confidence_from_decision_function(model, features):
    decision_values = model.decision_function(features)
    
    # Handle binary classification
    if decision_values.ndim == 1:
        # Apply the sigmoid function to map to [0,1]
        confidence = 1 / (1 + np.exp(-decision_values))
        probabilities = np.vstack([1 - confidence, confidence]).T
    else:
        # For multi-class classification, apply softmax
        # Softmax converts to probabilities that sum to 1 for each sample
        exp_values = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    return probabilities


def collect_probabilities(ml_models, user_input):
    model_probabilities = {}
    for model_name, model in ml_models.items():
        _, probabilities = predict_with_ml_model(model, user_input, extract_features)
        model_probabilities[model_name] = probabilities[0]
        result_text = "not sarcastic 🔴" if np.argmax(probabilities[0]) == 0 else "sarcastic 🟢"
        results[model_name] = result_text
    return model_probabilities

# Updated function to plot all model probabilities in one chart
def plot_combined_probability_chart(model_probabilities):
    labels = ["Not Sarcastic", "Sarcastic"]
    fig1 = go.Figure()
    for model_name, probabilities in model_probabilities.items():
        for i, label in enumerate(labels):
            fig1.add_trace(go.Bar(name=model_name + ' - ' + label, x=[model_name], y=[probabilities[i]], width=0.5))

    fig1.update_layout(
        barmode='group',
        title_text='Model Confidence Levels Comparison',
        xaxis_title="Model",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1])
    )

    for model_name, probabilities in model_probabilities.items():
        fig = go.Figure()

        # Gauge settings
        gauge_settings = lambda value, color: {
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': color},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.5], 'color': 'lightgray'},
                {'range': [0.5, 1], 'color': 'lightgray'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }

        # Animation frames
        frames = [
            go.Frame(
                data=[
                    go.Indicator(
                        mode="gauge+number",
                        value=probabilities[0],
                        domain={'x': [0, 0.48], 'y': [0, 1]},
                        gauge=gauge_settings(probabilities[0], "red"),
                        title="Not Sarcastic"
                    ),
                    go.Indicator(
                        mode="gauge+number",
                        value=probabilities[1],
                        domain={'x': [0.52, 1], 'y': [0, 1]},
                        gauge=gauge_settings(probabilities[1], "green"),
                        title="Sarcastic"
                    )
                ],
                name=str(probabilities)
            )
        ]

        # Initial state (empty gauges)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={'x': [0, 0.48], 'y': [0, 1]},
            gauge=gauge_settings(0, "blue"),
            title="Not Sarcastic"
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={'x': [0.52, 1], 'y': [0, 1]},
            gauge=gauge_settings(0, "red"),
            title="Sarcastic"
        ))

        fig.frames = frames

        # Update layout for animation
        fig.update_layout(
            title_text=f'Confidence Levels for {model_name}',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]}]
            }],
            transition={'duration': 20000}
        )
        st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig1)


# Streamlit app
st.title("Arabic Sarcasm Detection")
user_input = st.text_area("Enter Arabic text to analyze for sarcasm:")
ml_models = {
    'KNN': load_pickle_model(r'KNearest_Neighbors.pkl'),
    'Linear SVC': load_pickle_model(r'Lin_Support_Vector_Class.pkl'),
    'Naive Bayes': load_pickle_model(r'Naive_Bayes.pkl'),
    'Ridge Classifier': load_pickle_model(r'Ridge_Classifier.pkl'),
    'Logistic Regression': load_pickle_model(r'Logistic_Regression.pkl')
}
results = {}
if st.button("Analyze"):
    model_probabilities = collect_probabilities(ml_models, user_input)
    # TensorFlow model prediction
    tf_probabilities = predict_with_tf_model(user_input)
    model_probabilities['AraBERT'] = tf_probabilities
    tf_sarcasm_flag = np.argmax(tf_probabilities)
    #plot_probability_chart(tf_probabilities, 'AraBERT Model')
    result_text = "not sarcastic 🔴" if tf_sarcasm_flag == 0 else "sarcastic 🟢"
    results['AraBERT'] = result_text
    
    plot_combined_probability_chart(model_probabilities)

        
    # Convert the results to a DataFrame and display as a table
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Prediction'])
    st.table(results_df)
