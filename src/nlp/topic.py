import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.nlp.language import detect_language

class Model:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.valid_topics = ['багаж, забытые вещи и специальные грузы',
                            'билеты и тарифы',
                            'другое',
                            'лояльность и сертификаты',
                            'обмены, возвраты и официальные документы',
                            'сервис и обслуживание пассажиров',
                            'технические и цифровые сервисы'
                            ]
        
    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        topic = self.model.config.id2label[logits.argmax().item()]
        if topic in self.valid_topics:
            return topic 
        else:
            return 'invalid'

ru_predictor = None
en_predictor = None

def detect_ru_topic(text):
    global ru_predictor
    if ru_predictor is None:
        ru_predictor = Model("dalture/s7-ru-topics")
    return ru_predictor.predict(text)

def detect_en_topic(text):
    return "TBA"

def detect_topic(text):
    lang = detect_language(text)
    if lang == "ru":
        return detect_ru_topic(text)
    elif lang == "en":
        return detect_en_topic(text)
    else:
        return "undefined"