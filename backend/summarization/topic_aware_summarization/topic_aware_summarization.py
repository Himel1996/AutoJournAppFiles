import numpy as np
import torch
from utils import preprocessing, nltk_utilities
from utils.sentence_transformer_utilities import SentTransfUtilities


class TopicAwareSummarization:
    def __init__(self):
        self.sentence_transformer_model = "paraphrase-multilingual-MiniLM-L12-v2"

    def text_to_sentences(self, data):
        list_sentences = [*nltk_utilities.NltkSegmentizer().segment_into_sentences(data)]
        return list_sentences

    def preprocess(self, list_sentences, sent_transf_model_utils_obj):
        list_sentences = [preprocessing.remove_patterns(x) for x in list_sentences]
        list_sentences_per_doc_embeddings = [sent_transf_model_utils_obj.get_embeddings(x) for x in list_sentences if
                                             len(x) > 0]
        return list_sentences_per_doc_embeddings, list_sentences

    def compute_similarity_matrix(self, list_sentences_per_doc_embeddings, list_sentences, sent_transf_model_utils_obj,
                                  topic_embeddings, topics_df):
        similarity_matrix = np.zeros((len(topic_embeddings), len(list_sentences_per_doc_embeddings)))
        for i, cluster_embedding in enumerate(topic_embeddings):
            cluster_embedding = torch.from_numpy(cluster_embedding)
            for j, sentence_embedding in enumerate(list_sentences_per_doc_embeddings):
                sentence_embedding = sentence_embedding.to(torch.float64)
                similarity_matrix[i][j] = sent_transf_model_utils_obj.compute_cosine_similarity(cluster_embedding,
                                                                                                sentence_embedding)

        list_index_topics_within_matrix = np.argmax(similarity_matrix, axis=0)
        dict_topic_sentences = dict()
        topic_labels = topics_df["Name"].values.tolist()
        for index_sentence, index_id_topic in enumerate(list_index_topics_within_matrix):
            label_class = topic_labels[index_id_topic]
            if label_class not in dict_topic_sentences.keys():
                dict_topic_sentences[label_class] = list()
            dict_topic_sentences[label_class].append(list_sentences[index_sentence])

        return dict_topic_sentences

    def extract_topic_sentences(self, conv_doc, topics_df, topic_embeddings):
        sent_transf_model_utils_obj = SentTransfUtilities(model_name=self.sentence_transformer_model)
        # Topic sentence matrix for each conversation.
        data_sentences = self.text_to_sentences(conv_doc['dialogue'])
        sentence_embed, list_sentences = self.preprocess(data_sentences, sent_transf_model_utils_obj)
        dict_topic_sentences = self.compute_similarity_matrix(sentence_embed, list_sentences,
                                                              sent_transf_model_utils_obj, topic_embeddings,
                                                              topics_df)
        return dict_topic_sentences
