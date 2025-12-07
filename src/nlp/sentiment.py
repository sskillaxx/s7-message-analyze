from .language import detect_language

def detect_ru_sentiment(text):
    pass

def detect_en_sentiment(text):
    pass

def detect_sentiment(text):
    lang = detect_language(text)
    if lang == "ru":
        return detect_ru_sentiment(text)
    elif lang == "en":
        return detect_en_sentiment(text)
    else:
        return "undefined"