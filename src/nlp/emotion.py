import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from dotenv import load_dotenv
from src.nlp.language import detect_language
#import time

load_dotenv()
# os.environ['HTTP_PROXY'] = os.getenv("HTTP_PROXY")
# os.environ['HTTPS_PROXY'] = os.getenv("HTTPS_PROXY")

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
    #print('ru_em: начало обработки')
    
    #load_start = time.perf_counter()
    ru_predictor = Model("dalture/s7-ru-emotions")
    #load_time = time.perf_counter() - load_start
    #print(f'ru_em: модель загружена, время загрузки: {load_time:.4f}')
    
    #predict_start = time.perf_counter()
    #result = ru_predictor.predict(text)
    #predict_time = time.perf_counter() - predict_start
    #print(f"ru_em: предсказание сделано, затраченное время: {predict_time:.4f} сек")
    return ru_predictor.predict(text)

def detect_en_emotion(text):
    global en_predictor
    #print('en_em: начало обработки')
    
    #load_start = time.perf_counter()
    en_predictor = Model("dalture/s7-eng-emotions")
    #load_time = time.perf_counter() - load_start
    #print(f"en_em: модель загружена, время загрузки: {load_time:.4f} сек")
    
    #predict_start = time.perf_counter()
    #result = en_predictor.predict(text)
    #predict_time = time.perf_counter() - predict_start
    #print(f"en_em: предсказание сделано, затраченное время: {predict_time:.4f} сек")
    return en_predictor.predict(text)

def detect_emotion(text):
    lang = detect_language(text)
    if lang == "ru":
        return detect_ru_emotion(text)
    elif lang == "en":
        return detect_en_emotion(text)
    else:
        return "undefined"