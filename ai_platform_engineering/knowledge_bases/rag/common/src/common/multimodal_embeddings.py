"""
Multimodal embeddings for the RAG system.

This module wraps Cisco LiteLLM / Bedrock Nova multimodal embeddings. The
corpus-side ingestion path embeds image URLs discovered by the web loader, while
query-side tooling can embed local image files with the same model for live
image similarity search.
"""
import base64
import os
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import requests

try:
  from langchain_core.embeddings import Embeddings
except ImportError:
  class Embeddings:
    """Fallback so standalone image-search tooling can load without LangChain."""

try:
  from common.utils import get_logger
except ImportError:
  import logging

  def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

logger = get_logger(__name__)

NOVA_MULTIMODAL_MODEL_ID = "bedrock/amazon.nova-2-multimodal-embeddings-v1:0"
SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "webp"}


class ImageDownloadError(Exception):
  """Raised when an image URL cannot be downloaded."""


class ImageReadError(Exception):
  """Raised when a local image file cannot be read."""


class UnsupportedImageFormatError(Exception):
  """Raised when an image format is not supported by the embedding model."""


class NovaMultimodalEmbedder:
  """Client for Nova multimodal image embeddings through Cisco LiteLLM."""

  def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None):
    self.api_base = api_base or os.getenv("LITELLM_API_BASE")
    self.api_key = api_key or os.getenv("LITELLM_API_KEY")

    if not self.api_base:
      raise ValueError("LITELLM_API_BASE must be set for multimodal embeddings")
    if not self.api_key:
      raise ValueError("LITELLM_API_KEY must be set for multimodal embeddings")

    self.api_base = self.api_base.rstrip("/")

  def _detect_format_from_name(self, name: str, source_label: str) -> str:
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in SUPPORTED_IMAGE_FORMATS:
      raise UnsupportedImageFormatError(
        f"Unsupported or missing image format for {source_label!r}. "
        f"Supported formats: {sorted(SUPPORTED_IMAGE_FORMATS)}"
      )
    return "jpeg" if extension == "jpg" else extension

  def _detect_format(self, url: str) -> str:
    """Detect image format from a URL path."""
    path = urlparse(url).path
    return self._detect_format_from_name(path, url)

  def _download_image(self, url: str, timeout: int = 15) -> bytes:
    try:
      response = requests.get(url, timeout=timeout)
      response.raise_for_status()
      return response.content
    except requests.RequestException as e:
      raise ImageDownloadError(f"Failed to download image {url}: {e}") from e

  def _embed_image_bytes(self, image_bytes: bytes, source_label: str) -> List[float]:
    if not image_bytes:
      raise ImageReadError(f"Image source {source_label!r} is empty")

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
      response = requests.post(
        f"{self.api_base}/embeddings",
        headers={
          "Authorization": f"Bearer {self.api_key}",
          "Content-Type": "application/json",
        },
        json={
          "model": NOVA_MULTIMODAL_MODEL_ID,
          "input": encoded_image,
          "encoding_format": "base64",
        },
        timeout=60,
      )
      response.raise_for_status()
      response_body = response.json()
      embedding = response_body["data"][0]["embedding"]
      logger.info(
        "Generated multimodal embedding for %s with dimension %s",
        source_label,
        len(embedding),
      )
      return embedding
    except requests.RequestException as e:
      raise RuntimeError(f"LiteLLM proxy request failed for {source_label}: {e}") from e
    except (KeyError, IndexError) as e:
      raise RuntimeError(f"Unexpected response format from LiteLLM proxy: {e}") from e

  def embed_image_url(self, url: str) -> List[float]:
    """Generate an embedding for an image URL."""
    self._detect_format(url)
    image_bytes = self._download_image(url)
    return self._embed_image_bytes(image_bytes, url)

  def embed_image_path(self, path: Union[str, Path]) -> List[float]:
    """Generate an embedding for a local image file."""
    image_path = Path(path)
    self._detect_format_from_name(image_path.name, str(image_path))
    if not image_path.is_file():
      raise ImageReadError(f"Image file does not exist: {image_path}")
    try:
      image_bytes = image_path.read_bytes()
    except OSError as e:
      raise ImageReadError(f"Failed to read image file {image_path}: {e}") from e
    return self._embed_image_bytes(image_bytes, str(image_path))

  def embed_image_bytes(
    self,
    image_bytes: bytes,
    image_format: Optional[str] = None,
    source_label: str = "raw image",
  ) -> List[float]:
    """Generate an embedding for raw image bytes.

    image_format is optional because the LiteLLM request sends base64 image
    content. When provided, it is validated against the formats supported by
    Nova multimodal embeddings.
    """
    if image_format:
      normalized = image_format.lower().lstrip(".")
      self._detect_format_from_name(f"image.{normalized}", source_label)
    return self._embed_image_bytes(image_bytes, source_label)


class NovaMultimodalEmbeddingsAdapter(Embeddings):
  """LangChain Embeddings adapter for image URL embeddings."""

  def __init__(self, embedder: Optional[NovaMultimodalEmbedder] = None):
    self.embedder = embedder or NovaMultimodalEmbedder()

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Embed image URLs stored as document text."""
    return [self.embedder.embed_image_url(url) for url in texts]

  def embed_query(self, text: str) -> List[float]:
    """Embed a query image URL."""
    return self.embedder.embed_image_url(text)
