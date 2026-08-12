"""
Multimodal image embedding providers (Nova, Gemini) for ingestion and retrieval.
"""
import base64
import os
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import requests

from langchain_core.embeddings import Embeddings

from common.utils import get_logger

logger = get_logger(__name__)

SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "webp"}

# Model registry: provider name -> (model id on the embeddings proxy, output dimension)
_PROVIDER_REGISTRY = {
  "nova": (os.getenv("NOVA_MULTIMODAL_MODEL_ID", "bedrock/amazon.nova-2-multimodal-embeddings-v1:0"), 3072),
  "gemini": (os.getenv("GEMINI_MULTIMODAL_MODEL_ID", "vertex_ai/gemini-embedding-2"), 3072),
}


class ImageDownloadError(Exception):
  """Image could not be downloaded."""
  pass


class ImageReadError(Exception):
  """A local query image could not be read."""


class UnsupportedImageFormatError(Exception):
  """Image format is missing or not supported."""
  pass


def _detect_format(url: str) -> str:
  """Detect image format from the URL's file extension."""
  path = urlparse(url).path
  extension = path.rsplit(".", 1)[-1].lower() if "." in path else ""
  if extension not in SUPPORTED_IMAGE_FORMATS:
    raise UnsupportedImageFormatError(f"Unsupported or missing image format for URL: {url} (extension: '{extension}'). Supported: {sorted(SUPPORTED_IMAGE_FORMATS)}")
  return "jpeg" if extension == "jpg" else extension


def _download_image(url: str, timeout: int = 15) -> bytes:
  """Download raw image bytes from a URL."""
  try:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "CAIPE-Ingestor/1.0"})
    response.raise_for_status()
    return response.content
  except requests.RequestException as e:
    raise ImageDownloadError(f"Failed to download image from {url}: {e}") from e


class BaseMultimodalEmbedder:
  """Shared request/error handling for multimodal embedding providers."""

  provider_name: str = "base"
  model_id: str = ""
  dimension: int = 0

  def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None):
    self.api_base = api_base or os.getenv("LITELLM_API_BASE")
    self.api_key = api_key or os.getenv("LITELLM_API_KEY")

    if not self.api_base:
      raise ValueError("LITELLM_API_BASE environment variable is required")
    if not self.api_key:
      raise ValueError("LITELLM_API_KEY environment variable is required")

  def _build_payload(self, encoded_image: str, image_format: str) -> dict:
    raise NotImplementedError

  def _build_text_payload(self, text: str) -> dict:
    return {"model": self.model_id, "input": [text]}

  def _embed_image_bytes(self, image_bytes: bytes, image_format: str, source: str) -> List[float]:
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    payload = self._build_payload(encoded_image, image_format)

    try:
      response = requests.post(
        f"{self.api_base}/embeddings",
        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
      )
      response.raise_for_status()
      embedding = response.json()["data"][0]["embedding"]
      logger.info(f"Embedded image via {self.provider_name}: source={source}, dimension={len(embedding)}")
      return embedding
    except requests.RequestException as e:
      raise RuntimeError(f"Embeddings proxy request failed for image {source}: {e}") from e
    except (KeyError, IndexError) as e:
      raise RuntimeError(f"Unexpected response format from {self.provider_name} for {source}: {e}") from e

  def embed_image_url(self, url: str) -> List[float]:
    """Download an image and return its embedding vector."""
    image_format = _detect_format(url)
    image_bytes = _download_image(url)
    return self._embed_image_bytes(image_bytes, image_format, url)

  def embed_image_path(self, path: Union[str, Path]) -> List[float]:
    """Read a local query image and return its embedding vector."""
    image_path = Path(path)
    image_format = _detect_format(image_path.name)
    try:
      image_bytes = image_path.read_bytes()
    except OSError as e:
      raise ImageReadError(f"Failed to read image from {image_path}: {e}") from e
    return self._embed_image_bytes(image_bytes, image_format, str(image_path))

  def embed_text(self, text: str) -> List[float]:
    """Generate a provider-compatible text-to-image retrieval embedding."""
    if not text.strip():
      raise ValueError("Text query must not be empty")
    try:
      response = requests.post(
        f"{self.api_base}/embeddings",
        headers={
          "Authorization": f"Bearer {self.api_key}",
          "Content-Type": "application/json",
        },
        json=self._build_text_payload(text),
        timeout=60,
      )
      response.raise_for_status()
      embedding = response.json()["data"][0]["embedding"]
      logger.info(
        "Generated multimodal text embedding with dimension %s",
        len(embedding),
      )
      return embedding
    except requests.RequestException as e:
      raise RuntimeError(f"LiteLLM proxy text embedding request failed: {e}") from e
    except (KeyError, IndexError) as e:
      raise RuntimeError(f"Unexpected response format from LiteLLM proxy: {e}") from e


class NovaMultimodalEmbedder(BaseMultimodalEmbedder):
  """Embeds images using Amazon's Nova 2 Multimodal Embeddings model. Manual fallback option; Gemini is the default."""

  provider_name = "Nova"
  model_id, dimension = _PROVIDER_REGISTRY["nova"]

  def _build_payload(self, encoded_image: str, image_format: str) -> dict:
    return {"model": self.model_id, "input": encoded_image, "encoding_format": "base64"}

  def _build_text_payload(self, text: str) -> dict:
    return {"model": self.model_id, "input": text, "embeddingPurpose": "IMAGE_RETRIEVAL"}


class GeminiMultimodalEmbedder(BaseMultimodalEmbedder):
  """Embeds images using Google's Gemini Embedding 2 model. Default provider."""

  provider_name = "Gemini"
  model_id, dimension = _PROVIDER_REGISTRY["gemini"]

  def _build_payload(self, encoded_image: str, image_format: str) -> dict:
    mime_type = f"image/{image_format}"
    data_uri = f"data:{mime_type};base64,{encoded_image}"
    return {"model": self.model_id, "input": [data_uri]}


class MultimodalEmbeddingsFactory:
  """Selects a multimodal embedder. Defaults to Gemini; Nova available as a manual override."""
  _EMBEDDERS = {"nova": NovaMultimodalEmbedder, "gemini": GeminiMultimodalEmbedder}

  @classmethod
  def _get_provider_name(cls) -> str:
    explicit = os.getenv("MULTIMODAL_EMBEDDINGS_PROVIDER")
    if explicit:
      provider = explicit.lower()
      if provider not in cls._EMBEDDERS:
        raise ValueError(f"Unsupported multimodal embeddings provider: '{provider}'. Supported: {sorted(cls._EMBEDDERS)}")
      return provider

        # No explicit override: follow the text embedding model, when possible.
    from common.embeddings_factory import EmbeddingsFactory
    text_identifier = EmbeddingsFactory.get_provider_identifier().lower()
    for provider_name in cls._EMBEDDERS:
      if provider_name in text_identifier:
        return provider_name

    # Text model has no image-capable equivalent . Default to
    # gemini rather than block startup; incompatible embed attempts fail
    # individually and get logged, same as any other embed failure.
    return "gemini"
  @classmethod
  def get_embedder(cls) -> BaseMultimodalEmbedder:
    return cls._EMBEDDERS[cls._get_provider_name()]()

  @classmethod
  def get_embedding_dimension(cls) -> int:
    return cls._EMBEDDERS[cls._get_provider_name()].dimension


class MultimodalEmbeddingsAdapter(Embeddings):
  """Adapts a multimodal embedder to LangChain's Embeddings interface."""

  def __init__(self, embedder: Optional[BaseMultimodalEmbedder] = None):
    self.embedder = embedder or MultimodalEmbeddingsFactory.get_embedder()

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    return [self.embedder.embed_image_url(url) for url in texts]

  def embed_query(self, text: str) -> List[float]:
    return self.embedder.embed_image_url(text)


# Backward-compatible alias for existing imports (e.g. restapi.py)
NovaMultimodalEmbeddingsAdapter = MultimodalEmbeddingsAdapter
