"""Text embeddings from a Model2Vec static model, as an OPM text embeddings plugin."""
from typing import Any, Dict, List, Optional

import numpy as np
from ovos_plugin_manager.templates.embeddings import EmbeddingsArray, TextEmbedder

from ovos_m2v_pipeline import load_shared_model

DEFAULT_EMBEDDINGS_MODEL = "minishlab/potion-multilingual-128M"


class Model2VecEmbeddings(TextEmbedder):
    """One vector per text from a Model2Vec static model.

    Config:

    ``model`` : str (default: ``minishlab/potion-multilingual-128M``)
        A Hugging Face model id or a local model directory. The model is loaded once per process and shared with
        the intent pipelines that load the same path.
    ``normalize`` : bool (default: ``True``)
        Scale each vector to unit length, so a dot product is the cosine similarity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_path: str = self.config.get("model") or DEFAULT_EMBEDDINGS_MODEL
        self.normalize: bool = self.config.get("normalize", True)

    def get_embeddings(self, text: str) -> EmbeddingsArray:
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[EmbeddingsArray]:
        """All the texts in one ``encode`` call, one row per text in order."""
        if not texts:
            return []
        vectors = np.asarray(load_shared_model(self.model_path, "prototype").encode(list(texts)), dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        return list(vectors)
