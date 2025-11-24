import re
import spacy


def remove_patterns(text):
    """
        Remove punctions, emails, hashtags in given text
    """

    if isinstance(text, spacy.tokens.span.Span):
        text = text.text
    # Remove return char
    text = re.sub(r'\n', ' ', text)
    # Remove emails
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    return text


def preprocess(text):
    # Remove links
    text = re.sub(r"http\S+", "", text)
    # Remove tags
    text = re.sub(r"@\S+", "", text)  # remove tags
    # Encode and decode in order to UTF-16 in order to process emojis.
    text = text.encode('UTF-16', 'surrogatepass').decode(encoding='UTF-16')
    # remove leading and trailing spaces.
    text = text.strip()
    return text
