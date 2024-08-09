import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import emoji
import re
import string
import time
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "\U00002B50-\U00002B55"  # Miscellaneous Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "]+",
    flags=re.UNICODE,
)


def clean(text):
    text = text.lower()  # нижний регистр
    text = re.sub(r"http\S+", " ", text)  # удаляем ссылки
    text = re.sub(r"@\w+", " ", text)  # удаляем упоминания пользователей
    text = re.sub(r"#\w+", " ", text)  # удаляем хэштеги
    text = re.sub(r"\d+", " ", text)  # удаляем числа
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"<.*?>", " ", text)  #
    text = re.sub(r"[️«»—]", " ", text)
    text = re.sub(r"[^а-яё ]", " ", text)
    text = text.lower()
    text = emoji_pattern.sub(r"", text)
    return text

def predict_class(text,model_to_embed, model_to_predict, tokenizer):
  start_time = time.time()
  text = clean(text)
  class_list = ['Крипта', 'Мода', 'Спорт', 'Технологии', 'Финансы']
  encoded_input = tokenizer(text, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
  encoded_input = {k: v.to(model_to_embed.device) for k, v in encoded_input.items()}

  with torch.no_grad():
      model_output = model_to_embed(**encoded_input)

  embeddings = model_output.last_hidden_state[:, 0, :]

  embeddings = torch.nn.functional.normalize(embeddings)
    
  embeddings_np = embeddings.cpu().numpy()

  pred_class = model_to_predict.predict(embeddings_np)
    
  pred_proba = model_to_predict.predict_proba(embeddings_np)
  confidence = np.max(pred_proba)
  end_time = time.time()
  elapsed_time = end_time - start_time

  return f'Predicted class: {class_list[pred_class[0]]}, Confidence: {confidence:.4f}, Time: {round(elapsed_time, 4)}c'