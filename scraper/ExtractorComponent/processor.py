import os
from dotenv import load_dotenv

# Global parameters
load_dotenv("chat_bot_config.env")
ALPHA = os.environ.get("ALPHA") # Weights between content-content and title-content
PERCENTILE = os.environ.get("PERCENTILE") # Min-threshold for an aggregate similarity score
N_CLUSTERS = os.environ.get("N_CLUSTERS") # No. predefined clusters in a corpus

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
        
    def find_top_k(self) -> np.ndarray:
        text_text_sim_matrix = cosine_similarity(self.text_embeddings)
        text_title_sim_matrix = cosine_similarity(self.text_embeddings, self.title_embeddings)

        mask = ~np.eye(text_text_sim_matrix, dtype=bool)
        row_sums = (text_text_sim_matrix * mask).sum(axis=1)
        row_counts = mask.sum(axis=1)
        text_text_avg_sim_matrix = row_sums / row_counts

        agg_sim_matrix = (
            ALPHA * text_text_avg_sim_matrix
            + (1 - ALPHA) * text_title_sim_matrix
        )
        threshold = np.percentile(agg_sim_matrix, PERCENTILE)
        self.selected_idx = np.where(agg_sim_matrix >= threshold)
    
    def cluster(self):
        self.find_top_k()
        relevant_text_nodes = self.text_embeddings[self.selected_idx]
