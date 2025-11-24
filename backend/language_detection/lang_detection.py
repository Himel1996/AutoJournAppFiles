from langdetect import detect


class LangDetect:

    def is_english(self, text):
        return detect(text) == "en"
