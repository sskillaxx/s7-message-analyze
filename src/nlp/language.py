import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from dotenv import load_dotenv
#import time

load_dotenv()
# os.environ['HTTP_PROXY'] = os.getenv("HTTP_PROXY")
# os.environ['HTTPS_PROXY'] = os.getenv("HTTPS_PROXY")

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, force_download=True)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
model.eval()

def detect_language(text):
    #start = time.perf_counter()
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    prob, idx = torch.max(probs, dim=1)
    
    lang = model.config.id2label[idx.item()]
    
    if lang not in ['en', 'ru']:
        #elapsed = time.perf_counter() - start
        #print(f"язык не определён, время выполнения: {elapsed:.4f} сек")
        return 'unidentified'
    else:
        #elapsed = time.perf_counter() - start
        #print(f"язык определён, время выполнения: {elapsed:.4f} сек")
        return lang