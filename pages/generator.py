import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import textwrap

st.title('GPT2 trained on tg chat')

model_directory = 'finetuned/'  # Directory where the model is located
model = GPT2LMHeadModel.from_pretrained(model_directory, use_safetensors=True)
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

def predict(text, max_len=100, num_beams=10, temperature=1.5, top_p=0.7):
    with torch.inference_mode():
        prompt = text
        prompt = tokenizer.encode(prompt, return_tensors='pt')
        out = model.generate(
            input_ids=prompt,
            max_length=max_len,
            num_beams=num_beams,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=1,
            num_return_sequences=1,
            ).cpu().numpy()

    return textwrap.fill(tokenizer.decode(out[0]))



prompt = st.text_input("Твоя фраза")
col = st.columns(4)
with col[0]:
    max_len = st.slider("Text len", 20, 200, 100)
with col[1]:
    num_beams = st.slider("Beams", 0.1, 1., 0.5)
with col[2]:
    temperature = st.slider("Temperature", 0.1, 0.9, 0.35)
with col[3]:
    top_p = st.slider("Top-p", 0.1, 1.0, 0.7)

    submit = st.button('Сгенерировать ответ')

if submit:
    if prompt:
        pred = predict(prompt, max_len=max_len, num_beams=int(num_beams * 20), temperature=(1-temperature) * 5, top_p=top_p)
        st.write(pred)
