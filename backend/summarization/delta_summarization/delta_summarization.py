from typing import Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io

from utils.preprocessing import preprocess


class DeltaSummarization:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def find_cosine_similarity(self, summary1: str, summary2: str):
        embedding1 = self.model.encode(summary1, convert_to_tensor=True)
        embedding2 = self.model.encode(summary2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return cosine_similarity

    def create_cosine_similarity_matrix(self, img_buffer, topic_summaries: Dict[str, str]):
        topics = list(topic_summaries.keys())
        topic_summaries = list(topic_summaries.items())
        summary_count = len(topic_summaries)
        cosine_similarity_matrix = np.zeros((summary_count, summary_count))
        for idx in range(summary_count):
            for jdx in range(summary_count):
                # Cosine similarity is 1 if the same topics are being compared.
                if idx == jdx:
                    cosine_similarity_matrix[idx][jdx] = 1.0
                else:
                    cosine_similarity = self.find_cosine_similarity(topic_summaries[idx][1], topic_summaries[jdx][1])
                    cosine_similarity_matrix[idx][jdx] = cosine_similarity

        plt.imshow(cosine_similarity_matrix, cmap='viridis')
        plt.colorbar(label='Matrix Values')
        n = len(topics)
        plt.xticks(np.arange(n), topics, rotation='vertical')
        plt.yticks(np.arange(n), topics)
        plt.xlabel('Topics')
        plt.ylabel('Topics')
        title = 'Cosine Similarity between Topics'
        plt.title(title)
        # Adjust the figure size and margins
        plt.gcf().set_size_inches(10, 8)
        plt.tight_layout()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        return img_buffer

    def create_2d_scatter_plot(self, img_buffer, topic_summaries: Dict[str, str],
                               dialogue: str, default_summary: str):
        pca = PCA(n_components=2)
        topics = list(topic_summaries.keys())
        topics.extend(['Original Dialogue', 'Default Summary'])
        summaries = list(topic_summaries.values())
        summaries.extend([dialogue, default_summary])
        summary_embeddings = self.model.encode(summaries)
        summary_embeddings_2d = pca.fit_transform(summary_embeddings)
        # Scatter plot
        plt.scatter(summary_embeddings_2d[:, 0], summary_embeddings_2d[:, 1], marker='x')
        for i, label in enumerate(topics):
            plt.text(summary_embeddings_2d[i, 0], summary_embeddings_2d[i, 1], label)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Topic Summaries Scatter Plot")
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        return img_buffer

    def send_plot(self, plot_type: str, topic_summaries: Dict[str, str],
                  dialogue: str, default_summary: str):
        # Dialouge must be preprocessed since it directly comes from the original conversation.
        dialogue = preprocess(dialogue)
        img_buffer = io.BytesIO()
        if plot_type == 'cosine_similarity':
            img_buffer = self.create_cosine_similarity_matrix(img_buffer, topic_summaries)
        elif plot_type == '2d_scatter_plot':
            img_buffer = self.create_2d_scatter_plot(img_buffer, topic_summaries, dialogue, default_summary)

        return img_buffer
