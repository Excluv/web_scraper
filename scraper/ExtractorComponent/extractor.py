import os
from dotenv import load_dotenv

# Global parameters
load_dotenv("chat_bot_config.env")
ALPHA = os.environ.get("ALPHA") # Weights between content-content and title-content
BETA = os.environ.get("BETA") # The ratio between the 1st and 2nd clusters to determine which nodes should be kept
PERCENTILE = os.environ.get("PERCENTILE") # Min-threshold for an aggregate similarity score
N_CLUSTERS = os.environ.get("N_CLUSTERS") # No. predefined clusters in a corpus

import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

from .utils import SphericalKMeans
from .llm import LlmFactory


class Preprocessor:
    def __init__(self, html: BeautifulSoup):
        self.html = html

    def retrieve_text(self) -> List[str]:
        all_text_nodes = list(self.html.find_all(string=True))
        return all_text_nodes

    def retrieve_title(self) -> List[str]:
        title = [tag.get_text() for tag in self.html.find_all("title")]
        return title


class Postprocessor:
    def __init__(self, clustering_algo: str = "k-means"):
        if clustering_algo == "k-means":
            self.kmeans = SphericalKMeans(n_clusters=2, max_iter=300, random_state=42)
        
    def find_top_k(self,
                   title_embeddings: np.ndarray[float],
                   text_embeddings: np.ndarray[float]) -> np.ndarray[int]:
        text_text_sim_matrix = cosine_similarity(text_embeddings)
        text_title_sim_matrix = cosine_similarity(text_embeddings, title_embeddings)

        mask = ~np.eye(text_text_sim_matrix, dtype=bool)
        row_sums = (text_text_sim_matrix * mask).sum(axis=1)
        row_counts = mask.sum(axis=1)
        text_text_avg_sim_matrix = row_sums / row_counts

        agg_sim_matrix = (
            ALPHA * text_text_avg_sim_matrix
            + (1 - ALPHA) * text_title_sim_matrix
        )
        threshold = np.percentile(agg_sim_matrix, PERCENTILE)
        top_k_idx = np.where(agg_sim_matrix >= threshold)

        return top_k_idx
    
    def cluster(self, 
                text_embeddings: np.ndarray[float],
                selected_idx: np.ndarray[int]) -> np.ndarray[float]:
        if selected_idx:
            relevant_text_nodes = text_embeddings[selected_idx]
            node_clusters = self.kmeans.fit_predict(relevant_text_nodes)
        else:
            node_clusters = self.kmeans.fit_predict(text_embeddings)

        return node_clusters
        
    def select_nodes(self,
                     title_embeddings: np.ndarray[float],
                     text_embeddings: np.ndarray[float]) -> np.ndarray[int]:
        top_k_idx = self.find_top_k(title_embeddings, text_embeddings)
        text_node_clusters = self.cluster(text_embeddings, top_k_idx)
        cluster_densities = text_node_clusters.sum(axis=0)
        group1, group2 = cluster_densities

        if group1 > BETA * group2:
            selected_node_idx = np.where(text_node_clusters[:, 0] == 1)[0]
        elif group2 > BETA * group1:
            selected_node_idx = np.where(text_node_clusters[:, 1] == 1)[0]
        else:
            selected_node_idx = text_node_clusters[0]

        return selected_node_idx


class ContentExtractor:
    def __init__(self, 
                 html: BeautifulSoup,
                 processor: str = None):
        self.document = html
        self.llm = LlmFactory.create(processor)
        self.pre = Preprocessor(html)
        self.post = Postprocessor()

    def extract_content(self) -> str:
        document_title = self.pre.retrieve_title()
        document_texts = self.pre.retrieve_text()
        title_embeddings = self.llm.embed(document_title)
        text_embeddings = self.llm.embed(document_texts)

        relevant_text_idx = self.post.select_nodes(title_embeddings, text_embeddings)
        relevant_texts = document_texts[relevant_text_idx]

        return relevant_texts
