import streamlit as st
import joblib
from transformers import AutoTokenizer, AutoModel
from sasha_funcs import predict_class
st.title('TG channels classifier')
st.subheader('Model: Bert + LogReg')

model_clf = joblib.load('models/logistic_regression_model.pkl')
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model_bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")


text = st.text_input("Text to classify")

if text:
    st.write(predict_class(text, model_bert, model_clf, tokenizer))

button = st.button('Show 2 components with Umap Decomposition')

if button:
    st.image('images/scatter_of_tg_channels.png', width=500)