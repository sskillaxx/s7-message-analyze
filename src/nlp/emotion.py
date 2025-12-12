import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from dotenv import load_dotenv
from language import detect_language

load_dotenv()
os.environ['HTTP_PROXY'] = os.getenv("HTTP_PROXY")
os.environ['HTTPS_PROXY'] = os.getenv("HTTPS_PROXY")

class Model:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.valid_emotions = ['interest', 'fear', 'anger', 'others', 'joy', 'others']
        
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
    ru_predictor = Model("dalture/s7-ru-emotions")
    return ru_predictor.predict(text)

def detect_en_emotion(text):
    global en_predictor
    en_predictor = Model("dalture/s7-eng-emotions")
    return en_predictor.predict(text)

def detect_emotion(text):
    lang = detect_language(text)
    if lang == "ru":
        return detect_ru_emotion(text)
    elif lang == "en":
        return detect_en_emotion(text)
    else:
        return "undefined"