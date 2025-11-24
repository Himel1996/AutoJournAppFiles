from abc import ABC, abstractmethod
import numpy as np


class TopicModeling(ABC):

    @abstractmethod
    def get_topics(self, docs):
        """
        run topic modeling model to get the topics associated to each document in docs with optional
        topics assignments probabilities.

        :param docs: List[String]: list of documents input to the topic modeling model.
        :param num_topics: Int: number of topics.
        """
        pass

    @abstractmethod
    def preprocess(self, tweet):
        """
        Preprocess tweet by removing tags, links, etc.,
        :param tweet: String: single tweet text
        :return: String: processed tweet text.
        """
        pass

    def __merge_tweets(self, data):
        """
        Parse the tweets conversations in SAMSum format ot documents.
        Each processed conversation is treated as a document.

        :param data: List[{conv_id:String -> tweets:[String]}]
        :return: <List[document:String], Dict{document:String -> conv_id:String}>
        """

        doc_conv_dict = {}  # Map each processed document to its conversation
        docs = []
        for conv_dict in data:
            conv_id = conv_dict['id']
            conv = conv_dict['dialogue'].split("\n")
            processed_conv = []
            for tweet in conv:
                processed_conv.append(self.preprocess(tweet))

            processed_doc = ''.join(processed_conv)  # flatten each conversation as one document

            # Some conversation after being processed may be similar. Since, we have to keep them unique to be able
            # to map them back to original conversations, therefore, we add a suffixed special characters.
            while processed_doc in doc_conv_dict:
                processed_doc += "."

            doc_conv_dict[processed_doc] = conv_id
            docs.append(processed_doc)

        return docs, doc_conv_dict

    def __flatten_tweets(self, data):
        """
        Parse the tweets conversations in SAMSum format ot documents. One processed tweet is treated as a document.
        :param data: List[{conv_id:String -> tweets:[String]}]
        :return: <List[document:String], Dict{tweet:String -> conv_id:String},
            Dict{processed_tweet:String -> origin_tweet:String}>.
        """
        tweet_conv_dict = {}  # Map each tweet to its conversation
        processed_origin_tweet_dict = {}  # map processed tweet to original one
        docs = []
        for conv_dict in data:
            conv_id = conv_dict['id']
            conv = conv_dict['dialogue'].split("\n")
            for tweet in conv:
                tweet_conv_dict[tweet] = conv_id
                t_tweet = self.preprocess(tweet)
                # Some tweets after being processed will be similar. Since, we have to keep them unique to be able
                # to map them back to original tweets, therefore, we add a suffixed special characters.
                while t_tweet in processed_origin_tweet_dict:
                    t_tweet += "."

                processed_origin_tweet_dict[t_tweet] = tweet
                docs.append(t_tweet)

        return docs, tweet_conv_dict, processed_origin_tweet_dict

    def run_tweet_topic_modeling(self, data):
        """
        Run topic modeling model on tweet conversations in SAMSum format. It treats each tweet as a document and
        determine a topic for each tweet. Then, we average the probabilities for all tweets associated to a single
        conversation to get the topic probabilities for each conversation.

        :param data: List[{conv_id:String -> tweets:[String]}]

        :return: <conv_topic_probs_df.to_json():String, topics_df.to_json():String>:
            1- conv_topic_probs_df: each row is for a conversation and each column represent a topic probability.
            2- topics_df: each row is a topic and its assigned name based on tf-idf (i.e., top terms
                indicating the topic)
        """

        docs, tweet_conv_dict, processed_origin_tweet_dict = self.__flatten_tweets(data)  # each tweet is a document
        topics_df, probs, topic_embeddings = self.get_topics(docs)
        conv_tweet_topic_prob_dict = {}
        for idx, tweet in enumerate(docs):
            t_probs = probs[idx]
            t_conv = tweet_conv_dict[processed_origin_tweet_dict[tweet]]

            if t_conv not in conv_tweet_topic_prob_dict:
                conv_tweet_topic_prob_dict[t_conv] = []

            conv_tweet_topic_prob_dict[t_conv].append(t_probs)

        conv_topic_probs_dict = {}
        for conv_id, tweets_tprobs in conv_tweet_topic_prob_dict.items():
            mat = np.stack(tweets_tprobs, axis=0)  # create np matrix from list[np array] for faster computations
            conv_tprobs = np.average(mat, axis=0)  # compute average <tweet, topic> probability for each topic
            conv_topic_probs_dict[conv_id] = conv_tprobs.tolist()

        topics_id_name_dict = {row["Topic"]: row["Name"] for index, row in topics_df.iterrows()}
        return conv_topic_probs_dict, topics_id_name_dict

    def run_con_topic_modeling(self, data):
        """
        Run topic modeling model on tweet conversations in SAMSum format. It treats each conversation as a document and
        determine a topic for each conversation.

        :param data: List[{conv_id:String -> tweets:[String]}]

        :return: <conv_topic_probs_df: DataFrame, topics_df DataFrame>:
            1- conv_topic_probs_df: each row is for a conversation and each column represent a topic probability.
            2- topics_df: each row is a topic and its assigned name based on tf-idf (i.e., top terms
                indicating the topic)
        """

        docs, doc_conv_dict = self.__merge_tweets(data)  # each conversation is a document
        topics_df, probs = self.get_topics(docs)

        conv_topic_probs_dict = {}
        for idx, d in enumerate(docs):
            conv_id = doc_conv_dict[d]
            conv_topic_probs_dict[conv_id] = probs[idx].tolist()

        topics_id_name_dict = {row["Topic"]: row["Name"] for index, row in topics_df.iterrows()}
        return conv_topic_probs_dict, topics_id_name_dict

    def get_topic_embeddings(self, data):
        docs, _, _ = self.__flatten_tweets(data)
        topics_df, _, topic_embeddings = self.get_topics(docs)
        return topics_df, topic_embeddings
