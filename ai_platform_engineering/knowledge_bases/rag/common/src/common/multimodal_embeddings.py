"""
Multimodal image embedding providers (Nova, Gemini) for ingestion and retrieval.
"""
import base64
import os
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urljoin, urlparse

import requests

from langchain_core.embeddings import Embeddings

from common.utils import get_logger, sanitize_url

logger = get_logger(__name__)

SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "webp"}
DEFAULT_MAX_IMAGE_DOWNLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_REDIRECTS = 5

# Model registry: provider name -> (model id on the embeddings proxy, output dimension)
_PROVIDER_REGISTRY = {
  "nova": (os.getenv("NOVA_MULTIMODAL_MODEL_ID", "bedrock/amazon.nova-2-multimodal-embeddings-v1:0"), 3072),
  "gemini": (os.getenv("GEMINI_MULTIMODAL_MODEL_ID", "vertex_ai/gemini-embedding-2"), 3072),
}


def _safe_source_label(source: str) -> str:
  """Remove credentials, query parameters, and fragments from logged URLs."""
  parsed = urlparse(source)
  if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    return source
  host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
  try:
    port = parsed.port
  except ValueError:
    port = None
  if port:
    host = f"{host}:{port}"
  return parsed._replace(netloc=host, query="", fragment="").geturl()


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
    raise UnsupportedImageFormatError(f"Unsupported or missing image format for URL: {_safe_source_label(url)} (extension: '{extension}'). Supported: {sorted(SUPPORTED_IMAGE_FORMATS)}")
  return "jpeg" if extension == "jpg" else extension


def _validate_image_url(url: str) -> str:
  """Validate an image URL before each outbound request."""
  parsed = urlparse(url)
  if parsed.username is not None or parsed.password is not None:
    raise ValueError("embedded URL credentials are not allowed")
  allow_non_public = os.getenv("ALLOW_NON_PUBLIC_IMAGE_URLS", "false").lower() == "true"
  return sanitize_url(url, allow_non_public_urls=allow_non_public)


def _download_image(url: str, timeout: int = 15) -> bytes:
  """Download bounded image bytes while validating every redirect target."""
  max_bytes = int(os.getenv("MAX_IMAGE_DOWNLOAD_BYTES", str(DEFAULT_MAX_IMAGE_DOWNLOAD_BYTES)))
  current_url = url
  try:
    for redirect_count in range(MAX_IMAGE_REDIRECTS + 1):
      current_url = _validate_image_url(current_url)
      response = requests.get(
        current_url,
        timeout=timeout,
        headers={"User-Agent": "CAIPE-Ingestor/1.0"},
        allow_redirects=False,
        stream=True,
      )
      try:
        if response.status_code in {301, 302, 303, 307, 308}:
          location = response.headers.get("Location")
          if not location:
            raise ImageDownloadError(f"Image redirect from {_safe_source_label(current_url)} did not include a Location header")
          if redirect_count == MAX_IMAGE_REDIRECTS:
            raise ImageDownloadError(f"Image download exceeded {MAX_IMAGE_REDIRECTS} redirects")
          current_url = urljoin(current_url, location)
          continue

        response.raise_for_status()
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None and int(declared_length) > max_bytes:
          raise ImageDownloadError(f"Image exceeds the {max_bytes} byte download limit")

        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
          if not chunk:
            continue
          downloaded += len(chunk)
          if downloaded > max_bytes:
            raise ImageDownloadError(f"Image exceeds the {max_bytes} byte download limit")
          chunks.append(chunk)
        return b"".join(chunks)
      finally:
        response.close()
  except ImageDownloadError:
    raise
  except requests.RequestException as e:
    raise ImageDownloadError(f"Failed to download image from {_safe_source_label(url)}: {type(e).__name__}") from e
  except (ValueError, OSError) as e:
    raise ImageDownloadError(f"Failed to download image from {_safe_source_label(url)}: {e}") from e

  raise ImageDownloadError(f"Failed to download image from {_safe_source_label(url)}")


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
      logger.info(f"Embedded image via {self.provider_name}: source={_safe_source_label(source)}, dimension={len(embedding)}")
      return embedding
    except requests.RequestException as e:
      raise RuntimeError(f"Embeddings proxy request failed for image {_safe_source_label(source)}: {type(e).__name__}") from e
    except (KeyError, IndexError) as e:
      raise RuntimeError(f"Unexpected response format from {self.provider_name} for {_safe_source_label(source)}: {type(e).__name__}") from e

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

    # Text model has no image-capable equivalent. Default to
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
