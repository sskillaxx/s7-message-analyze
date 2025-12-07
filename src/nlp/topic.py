from .language import detect_language

def detect_ru_topic(text):
    pass

def detect_en_topic(text):
    pass

def detect_topic(text):
    lang = detect_language(text)
    if lang == "ru":
        return detect_ru_topic(text)
    elif lang == "en":
        return detect_en_topic(text)
    else:
        return "undefined"