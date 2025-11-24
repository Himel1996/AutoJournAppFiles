from topic_modeling.topic_modeling import TopicModeling
from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import pandas as pd


class Bertopic(TopicModeling):
    """
    Topic Modeling using Bertopic: https://maartengr.github.io/BERTopic/tutorial/algorithm/algorithm.html

    Note: The probability distribution of topics for each document outputted by the model is not a true probability
    distribution (i.e., topic probabilities for one document sum to 1). It merely shows how confident BERTopic is that
    certain topics can be found in a document.
    """

    def __init__(self, num_topics: int):
        self.num_topics = num_topics
        self.topics_list = [
        "Sports",
        "Politics",
        "Beauty and Fashion",
        "Movies and Entertainment",
        "Music",
        "Technology",
        "Health and Fitness",
        "Travel and Tourism",
        "Food and Cooking",
        "Literature",
        "Science",
        "History",
        "Education",
        "Art and Culture",
        "Finance and Economy",
        "Personal Development",
        "Relationships and Family",
        "Home and Garden",
        "Automobiles",
        "Gaming"
        ]
        self.static_topics_df = None 
        self.static_topic_embeddings = None
        umap_model = UMAP(n_neighbors=15,
                          transform_seed=173,  # fix a seed to avoid randomization in UMAP (we use a prime number)
                          n_components=5,
                          min_dist=0.0,
                          metric='cosine')
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = BERTopic(nr_topics=None,
                              language="multilingual",  # Use multilingual sentence-tranformers embedding model
                              top_n_words=5,
                              calculate_probabilities=True,
                              verbose=True,
                              n_gram_range=(1, 1),
                              umap_model=umap_model,
                              embedding_model=self.sentence_model)
        self.set_static_topics()

    def get_topics(self, docs):
        topics, probs = self.model.fit_transform(docs)  # fit the model to compute the topics
        # Reduce computed topics only if it's more than the given num_topics.
        if probs.shape[1] > self.num_topics:
            self.model.reduce_topics(docs, nr_topics=self.num_topics)
        topic_embeddings = self.model.topic_embeddings_
        # topic_df which hold in each row the topic number and name
        topics_df = self.model.get_topic_info()
        # remove outlier topic which has topic number = -1
        if -1 in topics_df["Topic"].tolist():
            topic_embeddings = topic_embeddings[1:]
        topics_df = topics_df[topics_df["Topic"] != -1]
        # new_probs has the same shape as probs. We will remove the columns of reduced topics (has zero probability)
        new_probs = np.apply_along_axis(lambda doc_prob: doc_prob[:len(topics_df)], axis=1, arr=probs)
        
        return topics_df, new_probs, topic_embeddings
    
    def set_static_topics(self):
        topics_df = pd.DataFrame(self.topics_list, columns=["Name"])
        topic_embeddings = self.sentence_model.encode(self.topics_list)
        topic_embeddings = np.array(topic_embeddings, dtype=np.float64)
        self.static_topics_df, self.static_topic_embeddings = topics_df, topic_embeddings

    def get_input_topic_embeddings(self, data):
        topics_df = pd.DataFrame(data, columns=["Name"])
        topic_embeddings = self.sentence_model.encode(data)
        topic_embeddings = np.array(topic_embeddings, dtype=np.float64)
        return topics_df, topic_embeddings

    def get_static_topics(self):
        return self.static_topics_df, self.static_topic_embeddings

    def preprocess(self, tweet):
        t_tweet = re.sub(r"http\S+", "", tweet)  # remove links
        t_tweet = re.sub(r"@\S+", "", t_tweet)  # remove tags

        # remove author which is at the beginning of each tweet delimtted by ':'
        t_tweet = re.sub(r"\w+:\s?", "", t_tweet)
        t_tweet = self.__remove_emojis(t_tweet)
        return t_tweet

    def __remove_emojis(self, tweet):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', tweet)

    def check_topic_count(self, num_topics):
        if self.num_topics != num_topics:
            self.num_topics = num_topics
