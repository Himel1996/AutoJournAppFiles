from typing import List

"""
Summarization models Interface
"""


class SummarizationModel(object):

    def preprocess(self, tweet: str) -> str:
        """
        Preprocess whole conversation by removing tags, links, etc.,
        :param tweet: String: single tweet text
        :return: String: processed tweet text.
        """
        pass

    def summarize(self, conv_tweets_list: List[str]) -> str:
        """
        Summarize the conversation given in the input,
        :param conv_tweets_list: List[str]: conversation list of tweets
        :return: String: summary of the input tweets
        """
        pass
