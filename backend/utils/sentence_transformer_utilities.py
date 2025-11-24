from sentence_transformers import SentenceTransformer, util


class SentTransfUtilities:
    model = None
    __model_name = None
    """
    Function: __init__
    Arguments:
        - model_name:
            Options:
                - 'all-MiniLM-L6-v2
                - 'nq-distilbert-base-v1'
                - 'paraphrase-multilingual-MiniLM-L12-v2'
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.__model_name = model_name
        if self.model == None:
            print("Initializing the Sentence Transformer model")
            self.model = SentenceTransformer(self.__model_name)

    """
    Function: get_embeddings()
    """

    def get_embeddings(self, src_data):
        return self.model.encode(src_data, convert_to_tensor=True, device='cpu')

    """
    Function: compute_cosine_similarity(query_embeddings, passage_embeddings)
    """

    def compute_cosine_similarity(self, query_embeddings, passage_embeddings):
        cosine_scores = util.cos_sim(query_embeddings, passage_embeddings)
        return cosine_scores
