import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

class Model:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.valid_emotions = {'interest', 'fear', 'anger', 'others', 'joy', 'surprise'}
        
    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        emotion = self.model.config.id2label[logits.argmax().item()]
        if emotion in self.valid_emotions:
            return emotion 
        else:
            return 'invalid'

ru_predictor = None
en_predictor = None

def detect_ru_emotion(text):
    global ru_predictor
    
    if ru_predictor is None:
        ru_predictor = Model("dalture/s7-ru-emotions")
        
    result = ru_predictor.predict(text)
    return result

def detect_en_emotion(text):
    global en_predictor
    
    if en_predictor is None:
        en_predictor = Model("dalture/s7-eng-emotions")
    
    result = en_predictor.predict(text)
    return result

def detect_emotion(text, language):
    if language == "ru":
        return detect_ru_emotion(text)
    elif language == "en":
        return detect_en_emotion(text)
    else:
        return "undefined"