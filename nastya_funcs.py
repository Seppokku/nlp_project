import time
import joblib
import re 
import string
import pymorphy3
import torch 
from transformers import BertModel, BertTokenizer
from torch import nn


model_name = "cointegrated/rubert-tiny2"
tokenizer = BertTokenizer.from_pretrained(model_name)

bert_model = BertModel.from_pretrained(model_name)


class MyTinyBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear = nn.Sequential(
            nn.Linear(312, 256),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(256, 6)
        )


    def forward(self, input_ids, attention_mask=None):
        # Pass the input_ids and attention_mask to the BERT model
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Normalize the output from BERT
        normed_bert_out = nn.functional.normalize(bert_out.last_hidden_state[:, 0, :])

        # Pass through the linear layer
        out = self.linear(normed_bert_out)

        return out
    

weights_path = "models/clf_rewievs_bert.pt"

model = MyTinyBERT()
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.to('cpu')
# tokenizer = transformers.AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2") 


# bert_model = transformers.AutoModel.from_pretrained("cointegrated/rubert-tiny2")
# weights_path = "./model_weights.pt"  # Replace with your .pt file path
# bert_model.load_state_dict(torch.load('models/clf_rewievs_bert.pt', map_location=torch.device('cpu')))

# bert_model.to('cpu')

morph = pymorphy3.MorphAnalyzer()

def lemmatize(text):
    words = text.split()
    lem_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lem_words)




logreg = joblib.load('models/logregmodel_new.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer_restaurants_new.pkl')

with open(
    "stopwords-ru.txt", "r", encoding="utf-8"
) as file:
    stop_words = set(file.read().split())


rating_dict = {
    1: "Отвратительно",
    2: "Плохо",
    3: "Удовлетворительно",
    4: "Хорошо",
    5: "Великолепно",}


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

def clean(text, stopwords):
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
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text


def predict_review(review):
    start_time = time.time()

    # Очистка и лемматизация текста
    clean_text = clean(review, stop_words)
    lem_text = lemmatize(clean_text)

    # Преобразование текста в TF-IDF представление
    X_new = vectorizer.transform([lem_text])

    # Предсказание
    prediction = logreg.predict(X_new)[0]

    # Проверка допустимости предсказания
    if prediction not in rating_dict:
        rating = "Ошибка предсказания"
    else:
        rating = rating_dict[prediction]

    # Измерение времени
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Лейбл: {prediction}")
    print(f"Оценка отзыва: {rating}")
    print(f"Затраченное время: {elapsed_time:.6f} seconds")
    return prediction, rating, elapsed_time


def preprocess_input(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    return inputs


def predict_bert(text):
    start_time = time.time()

    model.eval()
    inputs = preprocess_input(text)
    
    # Move tensors to the correct device if using GPU
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Since the output is already logits, no need to access outputs.logits
    predicted_class = outputs.argmax(dim=-1).item()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return predicted_class, rating_dict[predicted_class], elapsed_time