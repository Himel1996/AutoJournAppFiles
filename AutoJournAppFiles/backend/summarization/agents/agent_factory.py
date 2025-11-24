from config import Config
from summarization.agents import agent, base_agent, randomizer_agent, main_thread_agent, \
    conv_tree_agent, temporal_agent, social_media_agent
from summarization.models.summarizer import SummarizationModel


class AgentsFactory:

    @staticmethod
    def get_agent(summarizer_model: SummarizationModel) -> agent.Agent:
        method = Config.CONV_SUMMARIZER_METHOD

        if method == "telegram" or method == "reddit":
            return social_media_agent.SocialMediaAgent(summarizer_model=summarizer_model)

        if method == "tree":
            return conv_tree_agent.ConvTreeAgent(summarizer_model=summarizer_model)

        if method == "main_thread":
            return main_thread_agent.MainThreadAgent(summarizer_model=summarizer_model)

        if method == "random":
            return randomizer_agent.RandomizerAgent(summarizer_model=summarizer_model,
                                                    sample_size=Config.NUM_RANDOM_SAMPLES)

        if method == "temporal":
            time_bucket_len = Config.TIME_BUCKET_LEN_SEC
            if time_bucket_len == 0:
                time_bucket_len = 30 * 60  # default time bucket length is 30 minutes
            return temporal_agent.TemporalAgent(summarizer_model=summarizer_model,
                                                time_bucket_len_sec=time_bucket_len)

        if method == "base":
            return base_agent.BaseAgent(summarizer_model=summarizer_model)

        raise ModuleNotFoundError("No summarization method with the name: '" + method
                                  + "' Please use the defaults ones: tree, main_thread, random, temporal, and base")
