"""
Tests for NovaMultimodalEmbedder (multimodal_embeddings.py)
"""
import os
import pytest
import requests
from unittest.mock import patch, MagicMock

from common.multimodal_embeddings import (
  NovaMultimodalEmbedder,
  ImageDownloadError,
  UnsupportedImageFormatError,
)


class TestNovaMultimodalEmbedderInit:
  """Test suite for NovaMultimodalEmbedder initialization"""

  def test_reads_config_from_env(self):
    """Test that api_base and api_key are read from environment variables"""
    with patch.dict(os.environ, {"LITELLM_API_BASE": "https://proxy.example.com/v1", "LITELLM_API_KEY": "test-key"}):
      embedder = NovaMultimodalEmbedder()
      assert embedder.api_base == "https://proxy.example.com/v1"
      assert embedder.api_key == "test-key"

  def test_explicit_args_override_env(self):
    """Test that explicit constructor args take priority over env vars"""
    with patch.dict(os.environ, {"LITELLM_API_BASE": "https://proxy.example.com/v1", "LITELLM_API_KEY": "env-key"}):
      embedder = NovaMultimodalEmbedder(api_base="https://custom.example.com/v1", api_key="custom-key")
      assert embedder.api_base == "https://custom.example.com/v1"
      assert embedder.api_key == "custom-key"

  def test_missing_api_base_raises_value_error(self):
    """Test that a missing LITELLM_API_BASE raises ValueError"""
    with patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}, clear=True):
      with pytest.raises(ValueError, match="LITELLM_API_BASE"):
        NovaMultimodalEmbedder()

  def test_missing_api_key_raises_value_error(self):
    """Test that a missing LITELLM_API_KEY raises ValueError"""
    with patch.dict(os.environ, {"LITELLM_API_BASE": "https://proxy.example.com/v1"}, clear=True):
      with pytest.raises(ValueError, match="LITELLM_API_KEY"):
        NovaMultimodalEmbedder()


class TestDetectFormat:
  """Test suite for NovaMultimodalEmbedder._detect_format"""

  def _make_embedder(self):
    return NovaMultimodalEmbedder(api_base="https://proxy.example.com/v1", api_key="test-key")

  def test_detects_supported_formats(self):
    """Test that common supported image extensions are correctly detected"""
    embedder = self._make_embedder()
    assert embedder._detect_format("https://example.com/photo.jpg") == "jpeg"
    assert embedder._detect_format("https://example.com/photo.jpeg") == "jpeg"
    assert embedder._detect_format("https://example.com/logo.png") == "png"
    assert embedder._detect_format("https://example.com/anim.gif") == "gif"
    assert embedder._detect_format("https://example.com/pic.webp") == "webp"

  def test_missing_extension_raises_error(self):
    """Test that a URL with no file extension raises UnsupportedImageFormatError"""
    embedder = self._make_embedder()
    with pytest.raises(UnsupportedImageFormatError, match="Unsupported or missing image format"):
      embedder._detect_format("https://example.com/no-extension")

  def test_unsupported_extension_raises_error(self):
    """Test that an unsupported file extension raises UnsupportedImageFormatError"""
    embedder = self._make_embedder()
    with pytest.raises(UnsupportedImageFormatError, match="Unsupported or missing image format"):
      embedder._detect_format("https://example.com/document.pdf")


class TestDownloadImage:
  """Test suite for NovaMultimodalEmbedder._download_image"""

  def _make_embedder(self):
    return NovaMultimodalEmbedder(api_base="https://proxy.example.com/v1", api_key="test-key")

  def test_successful_download_returns_bytes(self):
    """Test that a successful download returns the response content bytes"""
    embedder = self._make_embedder()
    mock_response = MagicMock()
    mock_response.content = b"fake-image-bytes"
    mock_response.raise_for_status = MagicMock()
    with patch("common.multimodal_embeddings.requests.get", return_value=mock_response) as mock_get:
      result = embedder._download_image("https://example.com/photo.jpg")
      assert result == b"fake-image-bytes"
      mock_get.assert_called_once_with("https://example.com/photo.jpg", timeout=15)

  def test_network_failure_raises_image_download_error(self):
    """Test that a network failure during download raises ImageDownloadError"""
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get", side_effect=requests.ConnectionError("network down")):
      with pytest.raises(ImageDownloadError, match="Failed to download image"):
        embedder._download_image("https://example.com/photo.jpg")


class TestEmbedImageUrl:
  """Test suite for NovaMultimodalEmbedder.embed_image_url"""

  def _make_embedder(self):
    return NovaMultimodalEmbedder(api_base="https://proxy.example.com/v1", api_key="test-key")

  def test_successful_embedding_returns_vector(self):
    """Test the full successful flow: download, encode, call the proxy, return embedding"""
    embedder = self._make_embedder()

    mock_download_response = MagicMock()
    mock_download_response.content = b"fake-image-bytes"
    mock_download_response.raise_for_status = MagicMock()

    fake_embedding = [0.1, 0.2, 0.3]
    mock_proxy_response = MagicMock()
    mock_proxy_response.raise_for_status = MagicMock()
    mock_proxy_response.json.return_value = {"data": [{"embedding": fake_embedding}]}

    with patch("common.multimodal_embeddings.requests.get", return_value=mock_download_response):
      with patch("common.multimodal_embeddings.requests.post", return_value=mock_proxy_response) as mock_post:
        result = embedder.embed_image_url("https://example.com/photo.jpg")

    assert result == fake_embedding
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "bedrock/amazon.nova-2-multimodal-embeddings-v1:0"
    assert call_kwargs["json"]["encoding_format"] == "base64"
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

  def test_download_failure_propagates(self):
    """Test that a download failure during embed_image_url propagates as ImageDownloadError"""
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get", side_effect=requests.ConnectionError("network down")):
      with pytest.raises(ImageDownloadError):
        embedder.embed_image_url("https://example.com/photo.jpg")

  def test_unsupported_format_propagates(self):
    """Test that an unsupported image format raises before any download is attempted"""
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get") as mock_get:
      with pytest.raises(UnsupportedImageFormatError):
        embedder.embed_image_url("https://example.com/document.pdf")
      mock_get.assert_not_called()

  def test_proxy_error_response_raises_runtime_error(self):
    """Test that a failed proxy request (e.g. 400/500) raises RuntimeError"""
    embedder = self._make_embedder()

    mock_download_response = MagicMock()
    mock_download_response.content = b"fake-image-bytes"
    mock_download_response.raise_for_status = MagicMock()

    with patch("common.multimodal_embeddings.requests.get", return_value=mock_download_response):
      with patch("common.multimodal_embeddings.requests.post", side_effect=requests.HTTPError("400 Bad Request")):
        with pytest.raises(RuntimeError, match="LiteLLM proxy request failed"):
          embedder.embed_image_url("https://example.com/photo.jpg")

  def test_malformed_proxy_response_raises_runtime_error(self):
    """Test that an unexpected proxy response shape raises RuntimeError"""
    embedder = self._make_embedder()

    mock_download_response = MagicMock()
    mock_download_response.content = b"fake-image-bytes"
    mock_download_response.raise_for_status = MagicMock()

    mock_proxy_response = MagicMock()
    mock_proxy_response.raise_for_status = MagicMock()
    mock_proxy_response.json.return_value = {"unexpected": "shape"}

    with patch("common.multimodal_embeddings.requests.get", return_value=mock_download_response):
      with patch("common.multimodal_embeddings.requests.post", return_value=mock_proxy_response):
        with pytest.raises(RuntimeError, match="Unexpected response format"):
          embedder.embed_image_url("https://example.com/photo.jpg")
