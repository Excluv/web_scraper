import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity


class Preprocessor:
    def __init__(self, html: BeautifulSoup):
        self.html = html

    def process_content(self):
        all_text_nodes = list(self.html.find_all(string=True))


class Postprocessor:
    def __init__(self,
                 text_embeddings: np.ndarray[float],
                 title_embeddings: np.ndarray[float]):
        self.text_embeddings = text_embeddings
        self.title_embeddings = title_embeddings
        self.self_similarity_matrix = self._calc_similarity(text_embeddings)
        self.title_similarity_matrix = self._calc_similarity(text_embeddings, title_embeddings)
        self.mask = self._get_mask()

    def _get_mask(self):
        n_rows, n_cols = self.self_similarity_matrix.shape
        mask = []
        for i in range(n_rows):
            for j in range(n_cols):
                if i + j >= n_cols:
                    break
                if i == j:
                    continue
                mask.append([i, i + j])

        return mask

    def _calc_similarity(self,
                         X: np.ndarray[float], 
                         y: np.ndarray[float] = None):
        similarities = cosine_similarity(X, y)
        return similarities

    def cluster(self):
        pass
