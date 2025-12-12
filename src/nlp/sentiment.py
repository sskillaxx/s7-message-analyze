import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from pathlib import Path
from dotenv import load_dotenv
from .language import detect_language

load_dotenv()
os.environ['HTTP_PROXY'] = os.getenv("HTTP_PROXY")
os.environ['HTTPS_PROXY'] = os.getenv("HTTPS_PROXY")

SRC_DIR = Path(__file__).parent.parent
MODEL_DIR = SRC_DIR / "models"
class Model:
    def __init__(self, model_path: str):
        model_path = Path(model_path).resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
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
            return 'neutral'

ru_predictor = None
en_predictor = None

def detect_ru_sentiment(text):
    global ru_predictor
    ru_predictor = Model(MODEL_DIR / "ru_sentiment" / "checkpoint-215")
    return ru_predictor.predict(text)

def detect_en_sentiment(text):
    global en_predictor
    en_predictor = Model(MODEL_DIR / "eng_sentiment")
    return en_predictor.predict(text)

def detect_sentiment(text):
    lang = detect_language(text)
    if lang == "ru":
        return detect_ru_sentiment(text)
    elif lang == "en":
        return detect_en_sentiment(text)
    else:
        return "undefined"
    
if __name__ == "__main__":
    # Пример для теста
    print(detect_sentiment("где распечатать посадочный билет? на сайте есть ссылка, но она не рабочая( URI_ADDRESS"))
    print(detect_sentiment("what is prohibited to take in hand luggage?"))