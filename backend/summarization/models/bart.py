from transformers import pipeline, AutoTokenizer
from typing import List
from summarization.models.summarizer import SummarizationModel
import re
import logging


class Bart(SummarizationModel):
    def __init__(self, model):
        super(SummarizationModel, self).__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = pipeline("summarization", model=model)
        logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(funcName)s :: %(lineno)d \
        :: %(message)s', level=logging.INFO)
        self.log = logging.getLogger("bart")

    def preprocess(self, tweet):
        # Remove links
        tweet = re.sub(r"http\S+", "", tweet)
        # Remove tags
        tweet = re.sub(r"@\S+", "", tweet)  # remove tags
        # Encode and decode in order to UTF-16 in order to process emojis.
        tweet = tweet.encode('UTF-16', 'surrogatepass').decode(encoding='UTF-16')
        # remove leading and trailing spaces.
        tweet = tweet.strip()
        return tweet

    def __chunk_conversation(self, conv_tweets_list: List[str]) -> List[str]:
        """
        Takes a conversation which is a list of tweets. Since the BART model has limit on the input size,
        therefore, we have divide the tweets into chucks where each chunk has length = model max input lengthIt.

        The chunk is the concatenation of tweets where the total length <= model max input length.

        :param conv_tweets_list: List[tweets:[String]]
        :return: chunks: [String]
        """
        chunks = []
        chunk = ''
        length = 0

        for tweet in conv_tweets_list:
            tokenized_tweet = self._tokenizer.encode(tweet, truncation=False, max_length=None, return_tensors='pt')[0]

            if len(tokenized_tweet) > self._tokenizer.model_max_length:
                continue

            length += len(tokenized_tweet)

            if length <= self._tokenizer.model_max_length:  # append to current chunk
                chunk += tweet + "\n"  # in SAMSum format each tweet (i.e., conversation turn) should be in a new line
            else:  # create new chunk
                chunks.append(chunk)
                chunk = tweet
                length = len(tokenized_tweet)

        if len(chunk) > 0:  # check last chunk
            chunks.append(chunk)

        return chunks

    def __chunk_single_conversation(self, conversation: str, max_length=1024) -> str:
        chunks = []
        for i in range(0, len(conversation), max_length):
            chunk = conversation[i:i + max_length]
            chunks.append(chunk)
        return chunks

    def summarize(self, conv_tweets_list):
        conv_tweets_list = [self.preprocess(tweet) for tweet in conv_tweets_list]
        text_chunks = self.__chunk_single_conversation(conv_tweets_list[0], max_length=1024)
        chunk_summaries = []
        for i, chunk in enumerate(text_chunks):
            try:
                chunk_summary = self.model(chunk, min_length=1, max_length=128)
            except:
                continue
            chunk_summaries.append(chunk_summary)
            self.log.info("Summarized chunk number {}".format(i))

        chunks_summaries = [chunk_summary[0]["summary_text"] for chunk_summary in chunk_summaries]
        summary = ''.join(chunks_summaries)
        return summary
