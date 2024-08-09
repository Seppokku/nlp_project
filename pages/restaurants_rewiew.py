import streamlit as st
import joblib
from nastya_funcs import predict_review, predict_bert
st.title('Restaurant reviews classifier')


text = st.text_input("Text to classify")

if text:
    label, rating, time = predict_review(text)
    col = st.columns(2)
    col[0].write('Model: Tf-Idf + LogReg')
    col[0].write(f"Отзыв: {rating}({label})")
    col[0].write(f"Затраченное время: {time:.6f}с")

    cls_name, name, time1 = predict_bert(text)
    col[1].write('Model: Bert')
    col[1].write(f"Отзыв: {name}({cls_name})")
    col[1].write(f"Затраченное время: {time1:.6f}с")

