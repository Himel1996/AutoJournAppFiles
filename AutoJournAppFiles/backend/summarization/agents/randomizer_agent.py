from typing import Dict, List

from summarization.agents.base_agent import BaseAgent
from summarization.models.summarizer import SummarizationModel
import random


class RandomizerAgent(BaseAgent):
    def __init__(self, summarizer_model: SummarizationModel, sample_size: int):
        super(RandomizerAgent, self).__init__(summarizer_model)
        self.sample_size = sample_size

    def _sample(self, tweets: List[Dict]) -> List[Dict]:
        idx_set = set([])
        while len(idx_set) < self.sample_size:
            rand_idx = random.randrange(len(tweets))
            if rand_idx not in idx_set:
                idx_set.add(rand_idx)

        idx_set = sorted(idx_set)
        reservoir = [tweets[k] for k in idx_set]
        return reservoir

    def run_conv(self, conv_root: Dict) -> str:
        tweets_list = []
        self._flatten_tree(conv_root, tweets_list)
        sampled_tweets = self._sample(tweets_list)
        sampled_tweets = [tweet["username"] + ":" + tweet["text"] for tweet in sampled_tweets]
        summary = self.summarizer_model.summarize(sampled_tweets)
        return summary
