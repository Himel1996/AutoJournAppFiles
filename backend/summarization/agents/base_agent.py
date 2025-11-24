from typing import Dict, List

from summarization.agents.agent import Agent
from summarization.models.summarizer import SummarizationModel


class BaseAgent(Agent):

    def __init__(self, summarizer_model: SummarizationModel):
        super(BaseAgent, self).__init__(summarizer_model)

    def _flatten_tree(self, root: Dict, tweets_list: List[Dict]):
        tweets_list.append(root)

        # no replies
        if "replies" not in root or ("replies" in root and len(root["replies"]) == 0):
            return

        # Iterate over replies and recursively flatten it.
        for reply in root["replies"]:
            self._flatten_tree(reply, tweets_list)

    def run_conv(self, conv_root: Dict) -> str:
        tweets_list = []
        self._flatten_tree(conv_root, tweets_list)
        tweets_list.sort(key=lambda t: t["created_at"])  # sort tweets by date
        tweets_list = [tweet["username"] + ":" + tweet["text"] for tweet in tweets_list]
        summary = self.summarizer_model.summarize(tweets_list)
        return summary
