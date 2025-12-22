import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HTTP_PROXY'] = os.getenv("HTTP_PROXY")
os.environ['HTTPS_PROXY'] = os.getenv("HTTPS_PROXY")

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def detect_language(text):
    text = " ".join(text.strip().split())
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    prob, idx = torch.max(probs, dim=1)
    
    lang = model.config.id2label[idx.item()]
    
    if lang not in {'en', 'ru'}:
        return 'undefined'
    else:
        return lang