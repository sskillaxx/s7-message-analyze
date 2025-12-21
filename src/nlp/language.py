import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from dotenv import load_dotenv
import time

load_dotenv()
os.environ['HTTP_PROXY'] = os.getenv("HTTP_PROXY")
os.environ['HTTPS_PROXY'] = os.getenv("HTTPS_PROXY")

model_ckpt = "papluca/xlm-roberta-base-language-detection"
print('lang: начало обработки')
load_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
load_time = time.perf_counter() - load_start
print(f"lang: модель загружена, время загрузки: {load_time:.4f} сек")

def detect_language(text):
    start = time.perf_counter()
    text = " ".join(text.strip().split())
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    prob, idx = torch.max(probs, dim=1)
    
    lang = model.config.id2label[idx.item()]
    
    if lang not in {'en', 'ru'}:
        elapsed = time.perf_counter() - start
        print(f"язык не определён, время выполнения: {elapsed:.4f} сек")
        print(f"текст: '{text[:10]}'")
        return 'undefined'
    else:
        elapsed = time.perf_counter() - start
        print(f"язык определён, время выполнения: {elapsed:.4f} сек")
        print(f"текст: '{text[:10]}'")
        return lang