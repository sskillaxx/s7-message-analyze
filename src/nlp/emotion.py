from .language import detect_language

def detect_ru_emotion(text):
    pass

def detect_en_emotion(text):
    pass

def detect_emotion(text):
    lang = detect_language(text)
    if lang == "ru":
        return detect_ru_emotion(text)
    elif lang == "en":
        return detect_en_emotion(text)
    else:
        return "undefined"