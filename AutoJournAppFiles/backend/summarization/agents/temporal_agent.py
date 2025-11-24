from typing import Dict, List
from dateutil import parser

from summarization.agents.randomizer_agent import RandomizerAgent
from summarization.models.summarizer import SummarizationModel


class TemporalAgent(RandomizerAgent):
    def __init__(self, summarizer_model: SummarizationModel, time_bucket_len_sec: int):
        super(TemporalAgent, self).__init__(summarizer_model, sample_size=1)
        self.time_bucket_len_sec = time_bucket_len_sec

    @staticmethod
    def get_diff_sec(d1_str, d2_str):
        """
        Get difference between two string dates in seconds
        """
        dt1 = parser.parse(d1_str)
        dt2 = parser.parse(d2_str)
        delta = dt2 - dt1
        return delta.total_seconds()

    def __time_bucketing_conv(self, tweets: List[dict]) -> List[List[Dict]]:
        tweets.sort(key=lambda t: t["created_at"])  # sort tweets by date
        clusters = []
        idx = 0

        while idx < len(tweets):
            # create a new cluster
            head = tweets[idx]
            bucket_cluster = [head]
            idx += 1

            # add tweets in the current cluster
            while idx < len(tweets) and self.get_diff_sec(head["created_at"],
                                                          tweets[idx]["created_at"]) < self.time_bucket_len_sec:
                bucket_cluster.append(tweets[idx])
                idx += 1

            clusters.append(bucket_cluster)

        return clusters

    def run_conv(self, conv_root: Dict) -> str:
        tweets_list = []
        self._flatten_tree(conv_root, tweets_list)
        clusters = self.__time_bucketing_conv(tweets_list)
        sampled_tweets = [self._sample(temporal_cluster)[0] for temporal_cluster in clusters]
        sampled_tweets = [tweet["username"] + ":" + tweet["text"] for tweet in sampled_tweets]
        summary = self.summarizer_model.summarize(sampled_tweets)
        return summary
