"""
   Read configuration parameters from configuration file
"""

import json
import os


class Config:
    # Twitter API config
    CONSUMER_KEY = ""
    CONSUMER_SECRET = ""
    API_MAX_NUM_CONVERSATIONS = 0
    API_MAX_NUM_PAGES = 0
    API_MAX_PAGE_NUM_RESULTS = 0

    # Topic Modeling config
    TOPIC_PER_TWEET = False  # boolean for performing topic modeling on tweet level

    # Text Summarization config
    NUM_RANDOM_SAMPLES = 0  # size of sample from randomizer agent
    # implemented methods: tree, main_thread, random, temporal, base
    CONV_SUMMARIZER_METHOD = ''
    TIME_BUCKET_LEN_SEC = 0  # time bucket length for temporal agent summarizer
    SUMMARIZATION_MODEL = ''  # summarization model from hugging-face
    OPENAI_API_KEY = ''

    # Telegram API config
    TELEGRAM_API_ID = "22557261"
    TELEGRAM_API_HASH = "777fd1cf942701becad6259a61212e88"
    MAX_NUM_OF_TELEGRAM_CHANNELS = 0
    MAX_NUM_OF_TELEGRAM_MESSAGES_PER_CHANNEL = 0
    TELEGRAM_PHONE_NUMBER = ""
    TELEGRAM_PASSWORD = ""

    # Reddit API config
    REDDIT_API_ID = ""
    REDDIT_API_SECRET = ""
    REDDIT_USERNAME = ""
    REDDIT_PASSWORD = ""
    REDDIT_APP_NAME = ""


config_vars = {config: t for config, t in vars(
    Config).items() if not config.startswith('__')}


def _read_config(source):
    """
        Read configuration from source into a dictionary
    """

    result = {}
    for config, default in config_vars.items():
        t = type(default)
        if config in source:
            result[config] = t(source[config]) if t is not bool else str(
                source[config]).lower() == 'true'

    return result


def _get_file_config(testing=False):
    """
        Read configuration from file
    """

    file_name = 'config.json'

    with open(file_name) as f:
        file_config = json.load(f)
        return _read_config(file_config)


def _get_env_config():
    """
        Read configuration from defined env variables
    """
    return _read_config(os.environ)


def _get_config(config, *stages):
    for stage in stages:  # return the value of the first stage defining the value
        if config in stage:
            return stage[config]


def init():
    file_config = _get_file_config()  # read configuration from file
    # read configurations from environment (env variables)
    env_config = _get_env_config()

    # Define configurations as stages in the order of most priority (e.g., env variables has top priority)
    stages = env_config, file_config

    # for each configuration variable get its value from first stage defining it.
    for config in config_vars:
        val = _get_config(config, *stages)
        if val is not None:
            setattr(Config, config, val)
        else:
            raise ValueError('Missing config \'{}\''.format(config))
