import os
from dotenv import load_dotenv

load_dotenv("chat_bot_config.env")

import numpy as np
from google import genai


class LlmClient:
    def prompt(self, 
               texts: np.ndarray[str]) -> ...:
        raise NotImplementedError

    def embed(self, 
              texts: np.ndarray[str],
              task_type: str,
              dimensionality: int = 768) -> np.ndarray:
        raise NotImplementedError


class LocalClient(LlmClient):
    pass


class RemoteClient(LlmClient):
    def __init__(self, provider: str):
        self.provider = provider
        self.client = self._select_provider(provider)

    def _select_provider(self, provider: str) -> ...:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        if provider.lower() == "google":
            return genai.Client(api_key=api_key)
        else:
            raise ValueError("Invalid provider")

    def _select_model(self, provider: str, purpose: str) -> str:
        model_map = {
            "google": {
                "embedding": "gemini-embedding-001",
            },
        }
        return model_map[provider][purpose]

    def embed(self, 
              texts: np.ndarray[str], 
              task_type: str = "SEMANTIC_SIMILARITY",
              dimensionality: int = 768) -> np.ndarray[float]:
        model = self._select_model(self.provider, "embedding")
        embeddings = np.array([
            np.array(e.values) for e in self.client.embed_content(
                model=model,
                contents=texts,
                config=genai.types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=dimensionality    
                )
            ).embeddings
        ])
        normed_embedding = embeddings / np.linalg.norm(embeddings, axis=1, keep_dims=True)
        return normed_embedding
    

class LlmFactory:
    def create(self, provider: str):
        if provider == "local":
            return LocalClient()
        else:
            return RemoteClient(provider)
