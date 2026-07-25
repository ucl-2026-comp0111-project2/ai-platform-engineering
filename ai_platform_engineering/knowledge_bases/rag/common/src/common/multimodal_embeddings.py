"""
Multimodal Embeddings for RAG System.

This module provides image embedding support via Amazon's Nova 2
Multimodal Embeddings model, accessed through Cisco's LiteLLM proxy,
used for the pre-embedding ingestion pipeline.
this module downloads an image from a URL, base64-encodes it, and calls a multimodal embedding
model through the proxy's OpenAI-compatible /v1/embeddings endpoint.
"""
import base64
import os
from typing import List, Optional
from urllib.parse import urlparse

import requests
from langchain_core.embeddings import Embeddings

from common.utils import get_logger

logger = get_logger(__name__)

# Model ID as configured on the LiteLLM proxy
NOVA_MULTIMODAL_MODEL_ID = "bedrock/amazon.nova-2-multimodal-embeddings-v1:0"

# Supported image file extensions (informational; the model itself
# detects format from the decoded image bytes, not from this value)
SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "webp"}


class ImageDownloadError(Exception):
  """Raised when an image cannot be downloaded from its URL."""
  pass


class UnsupportedImageFormatError(Exception):
  """Raised when an image's format cannot be determined or is not supported."""
  pass


class NovaMultimodalEmbedder:
  """
  Embeds images using Amazon's Nova 2 Multimodal Embeddings model,
  accessed through Cisco's LiteLLM proxy.

  Environment Variables:
      LITELLM_API_BASE: Base URL of the LiteLLM proxy (required)
      LITELLM_API_KEY: API key for the proxy (required)
  """

  def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None):
    """
    Args:
        api_base: Base URL of the LiteLLM proxy. Defaults to the
            LITELLM_API_BASE environment variable.
        api_key: API key for the proxy. Defaults to the LITELLM_API_KEY
            environment variable.
    Raises:
        ValueError: If api_base or api_key are not provided and not set
            in the environment.
    """
    self.api_base = api_base or os.getenv("LITELLM_API_BASE")
    self.api_key = api_key or os.getenv("LITELLM_API_KEY")
    if not self.api_base:
      raise ValueError("LITELLM_API_BASE environment variable is required for NovaMultimodalEmbedder")
    if not self.api_key:
      raise ValueError("LITELLM_API_KEY environment variable is required for NovaMultimodalEmbedder")

  def _detect_format(self, url: str) -> str:
    """
    Detect the image format from a URL's file extension. Used only for
    validation before downloading; the model itself detects format from
    the actual image bytes.

    Args:
        url: The image URL.
    Returns:
        The detected format string (e.g. "jpeg", "png").
    Raises:
        UnsupportedImageFormatError: If the extension is missing or not supported.
    """
    path = urlparse(url).path
    extension = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if extension not in SUPPORTED_IMAGE_FORMATS:
      raise UnsupportedImageFormatError(f"Unsupported or missing image format for URL: {url} (extension: '{extension}'). Supported: {sorted(SUPPORTED_IMAGE_FORMATS)}")
    return "jpeg" if extension == "jpg" else extension

  def _download_image(self, url: str, timeout: int = 15) -> bytes:
    """
    Download raw image bytes from a URL.

    Args:
        url: The image URL.
        timeout: Request timeout in seconds.
    Returns:
        Raw image bytes.
    Raises:
        ImageDownloadError: If the download fails for any reason (network error,
            non-200 status, etc.).
    """
    try:
      response = requests.get(url, timeout=timeout)
      response.raise_for_status()
      return response.content
    except requests.RequestException as e:
      raise ImageDownloadError(f"Failed to download image from {url}: {e}") from e

  def embed_image_url(self, url: str) -> List[float]:
    """
    Download an image from a URL and generate its embedding vector via
    Cisco's LiteLLM proxy.

    Args:
        url: The image URL to embed.
    Returns:
        The embedding vector as a list of floats.
    Raises:
        ImageDownloadError: If the image cannot be downloaded.
        UnsupportedImageFormatError: If the image format is not supported.
        RuntimeError: If the proxy call fails or returns an unexpected response.
    """
    self._detect_format(url)
    image_bytes = self._download_image(url)
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
      response = requests.post(
        f"{self.api_base}/embeddings",
        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
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
      logger.info(f"Embedded image via LiteLLM proxy: url={url}, dimension={len(embedding)}")
      return embedding
    except requests.RequestException as e:
      raise RuntimeError(f"LiteLLM proxy request failed for image {url}: {e}") from e
    except (KeyError, IndexError) as e:
      raise RuntimeError(f"Unexpected response format from Nova embedding model for {url}: {e}") from e


class NovaMultimodalEmbeddingsAdapter(Embeddings):
  """
  Adapts NovaMultimodalEmbedder to LangChain's Embeddings interface,
  so it can be used with LangChain's Milvus vector store wrapper the
  same way text embedding providers are used in embeddings_factory.py.

  Note: "documents" and "query" here are image URLs, not raw text.
  """

  def __init__(self, embedder: Optional[NovaMultimodalEmbedder] = None):
    """
    Args:
        embedder: An existing NovaMultimodalEmbedder instance to use.
            If not provided, a new one is created with default settings.
    """
    self.embedder = embedder or NovaMultimodalEmbedder()

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """
    Embed a list of image URLs.

    Args:
        texts: A list of image URLs (despite the LangChain parameter name).
    Returns:
        A list of embedding vectors, one per URL.
    """
    return [self.embedder.embed_image_url(url) for url in texts]

  def embed_query(self, text: str) -> List[float]:
    """
    Embed a single image URL for querying.

    Args:
        text: An image URL (despite the LangChain parameter name).
    Returns:
        The embedding vector.
    """
    return self.embedder.embed_image_url(text)
