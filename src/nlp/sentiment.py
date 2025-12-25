import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

class Model:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.valid_sentiments = ['negative', 'neutral', 'positive']
        
    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        sentiment = self.model.config.id2label[logits.argmax().item()]
        if sentiment in self.valid_sentiments:
            return sentiment 
        else:
            return 'invalid'

ru_predictor = None
en_predictor = None

def detect_ru_sentiment(text):
    global ru_predictor
    
    if ru_predictor is None:
        ru_predictor = Model("dalture/s7-ru-sentiment")
    
    result = ru_predictor.predict(text)
    return result

def detect_en_sentiment(text):
    global en_predictor
    
    if en_predictor is None:
        ru_predictor = Model("dalture/s7-eng-sentiment")
    
    result = ru_predictor.predict(text)
    return result

def detect_sentiment(text, language):
    if language == "ru":
        return detect_ru_sentiment(text)
    elif language == "en":
        return detect_en_sentiment(text)
    else:
        return "undefined"