from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingGenerator:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """
        Initialize the embedding generator with the specified model.

        Args:
            model_name (str): Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the input text(s).

        Args:
            text (Union[str, List[str]]): Input text or list of texts

        Returns:
            np.ndarray: Embedding vector(s)
        """
        return self.model.encode(text, normalize_embeddings=True)


if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator()

    # Single text example
    text = "这是一个测试句子"
    embedding = generator.generate_embedding(text)
    print(f"Single text embedding shape: {embedding.shape}")

    # Multiple texts example
    texts = ["这是第一个句子", "这是第二个句子", "This is an English sentence"]
    embeddings = generator.generate_embedding(texts)
    print(f"Multiple texts embedding shape: {embeddings.shape}")

    # Calculate similarity between embeddings
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f"Similarity between first two sentences: {similarity:.4f}")

    similarity = np.dot(embeddings[0], embeddings[2])
    print(f"Similarity between first and third sentences: {similarity:.4f}")
