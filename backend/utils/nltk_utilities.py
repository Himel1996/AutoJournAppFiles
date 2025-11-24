import nltk
from nltk.tokenize import sent_tokenize
nltk.data.path.append("/nltk_data/")


class NltkSegmentizer:
    def __init__(self):
        print("Initializing NltkSegmentizer object")
        nltk.download('punkt')

    """
    Function: segment_into_sentences
    """
    def segment_into_sentences(self, src_text="", _format=""):
        intermediate_result = None

        if isinstance(src_text, str):
            intermediate_result = sent_tokenize(src_text)
        elif isinstance(src_text, list):
            intermediate_result = list()

            for sent in src_text:
                intermediate_result.extend(sent_tokenize(sent))

        return intermediate_result
