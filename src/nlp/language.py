import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, force_download=True)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
model.eval()

def detect_language(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    prob, idx = torch.max(probs, dim=1)
    
    lang = model.config.id2label[idx.item()]
    
    if lang not in ['en', 'ru']:
        return 'unidentified'
    else:
        return lang