"""Tests for multimodal_embeddings.py (Nova, Gemini embedders and the provider factory)."""
import os
import pytest
import requests
from unittest.mock import patch, MagicMock

from common.multimodal_embeddings import (
  NovaMultimodalEmbedder,
  GeminiMultimodalEmbedder,
  MultimodalEmbeddingsFactory,
  ImageDownloadError,
  ImageReadError,
  UnsupportedImageFormatError,
  _detect_format,
  _download_image,
)


ENV = {"LITELLM_API_BASE": "https://proxy.example.com/v1", "LITELLM_API_KEY": "test-key"}


class TestBaseEmbedderInit:
  """Init behavior is shared across providers (BaseMultimodalEmbedder), tested against both."""

  @pytest.mark.parametrize("embedder_class", [NovaMultimodalEmbedder, GeminiMultimodalEmbedder])
  def test_reads_config_from_env(self, embedder_class):
    with patch.dict(os.environ, ENV):
      embedder = embedder_class()
      assert embedder.api_base == "https://proxy.example.com/v1"
      assert embedder.api_key == "test-key"

  @pytest.mark.parametrize("embedder_class", [NovaMultimodalEmbedder, GeminiMultimodalEmbedder])
  def test_explicit_args_override_env(self, embedder_class):
    with patch.dict(os.environ, ENV):
      embedder = embedder_class(api_base="https://custom.example.com/v1", api_key="custom-key")
      assert embedder.api_base == "https://custom.example.com/v1"
      assert embedder.api_key == "custom-key"

  @pytest.mark.parametrize("embedder_class", [NovaMultimodalEmbedder, GeminiMultimodalEmbedder])
  def test_missing_api_base_raises_value_error(self, embedder_class):
    with patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}, clear=True):
      with pytest.raises(ValueError, match="LITELLM_API_BASE"):
        embedder_class()

  @pytest.mark.parametrize("embedder_class", [NovaMultimodalEmbedder, GeminiMultimodalEmbedder])
  def test_missing_api_key_raises_value_error(self, embedder_class):
    with patch.dict(os.environ, {"LITELLM_API_BASE": "https://proxy.example.com/v1"}, clear=True):
      with pytest.raises(ValueError, match="LITELLM_API_KEY"):
        embedder_class()


class TestDetectFormat:
  """Module-level format detection, shared by all providers."""

  def test_detects_supported_formats(self):
    assert _detect_format("https://example.com/photo.jpg") == "jpeg"
    assert _detect_format("https://example.com/photo.jpeg") == "jpeg"
    assert _detect_format("https://example.com/logo.png") == "png"
    assert _detect_format("https://example.com/anim.gif") == "gif"
    assert _detect_format("https://example.com/pic.webp") == "webp"

  def test_missing_extension_raises_error(self):
    with pytest.raises(UnsupportedImageFormatError, match="Unsupported or missing image format"):
      _detect_format("https://example.com/no-extension")

  def test_unsupported_extension_raises_error(self):
    with pytest.raises(UnsupportedImageFormatError, match="Unsupported or missing image format"):
      _detect_format("https://example.com/document.pdf")


class TestDownloadImage:
  """Module-level download, shared by all providers."""

  def test_successful_download_returns_bytes(self):
    mock_response = MagicMock()
    mock_response.content = b"fake-image-bytes"
    mock_response.raise_for_status = MagicMock()
    with patch("common.multimodal_embeddings.requests.get", return_value=mock_response) as mock_get:
      result = _download_image("https://example.com/photo.jpg")
      assert result == b"fake-image-bytes"
      mock_get.assert_called_once()
      assert mock_get.call_args.args == ("https://example.com/photo.jpg",)
      assert mock_get.call_args.kwargs["timeout"] == 15

  def test_network_failure_raises_image_download_error(self):
    with patch("common.multimodal_embeddings.requests.get", side_effect=requests.ConnectionError("network down")):
      with pytest.raises(ImageDownloadError, match="Failed to download image"):
        _download_image("https://example.com/photo.jpg")


def _mock_download(content: bytes = b"fake-image-bytes") -> MagicMock:
  response = MagicMock()
  response.content = content
  response.raise_for_status = MagicMock()
  return response


def _mock_proxy_response(embedding=None, body: dict = None) -> MagicMock:
  response = MagicMock()
  response.raise_for_status = MagicMock()
  response.json.return_value = body if body is not None else {"data": [{"embedding": embedding or [0.1, 0.2, 0.3]}]}
  return response


class TestNovaEmbedImageUrl:
  def _make_embedder(self):
    return NovaMultimodalEmbedder(api_base="https://proxy.example.com/v1", api_key="test-key")

  def test_successful_embedding_uses_raw_base64_payload(self):
    embedder = self._make_embedder()
    fake_embedding = [0.1, 0.2, 0.3]
    with patch("common.multimodal_embeddings.requests.get", return_value=_mock_download()):
      with patch("common.multimodal_embeddings.requests.post", return_value=_mock_proxy_response(fake_embedding)) as mock_post:
        result = embedder.embed_image_url("https://example.com/photo.jpg")

    assert result == fake_embedding
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == embedder.model_id
    assert call_kwargs["json"]["encoding_format"] == "base64"
    assert isinstance(call_kwargs["json"]["input"], str)  # raw base64 string, not wrapped
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

  def test_download_failure_propagates(self):
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get", side_effect=requests.ConnectionError("network down")):
      with pytest.raises(ImageDownloadError):
        embedder.embed_image_url("https://example.com/photo.jpg")

  def test_unsupported_format_raises_before_download(self):
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get") as mock_get:
      with pytest.raises(UnsupportedImageFormatError):
        embedder.embed_image_url("https://example.com/document.pdf")
      mock_get.assert_not_called()

  def test_proxy_error_response_raises_runtime_error(self):
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get", return_value=_mock_download()):
      with patch("common.multimodal_embeddings.requests.post", side_effect=requests.HTTPError("400 Bad Request")):
        with pytest.raises(RuntimeError, match="Embeddings proxy request failed"):
          embedder.embed_image_url("https://example.com/photo.jpg")

  def test_malformed_proxy_response_raises_runtime_error(self):
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get", return_value=_mock_download()):
      with patch("common.multimodal_embeddings.requests.post", return_value=_mock_proxy_response(body={"unexpected": "shape"})):
        with pytest.raises(RuntimeError, match="Unexpected response format"):
          embedder.embed_image_url("https://example.com/photo.jpg")


class TestGeminiEmbedImageUrl:
  def _make_embedder(self):
    return GeminiMultimodalEmbedder(api_base="https://proxy.example.com/v1", api_key="test-key")

  def test_successful_embedding_uses_data_uri_payload(self):
    embedder = self._make_embedder()
    fake_embedding = [0.4, 0.5, 0.6]
    with patch("common.multimodal_embeddings.requests.get", return_value=_mock_download()):
      with patch("common.multimodal_embeddings.requests.post", return_value=_mock_proxy_response(fake_embedding)) as mock_post:
        result = embedder.embed_image_url("https://example.com/photo.jpg")

    assert result == fake_embedding
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == embedder.model_id
    assert "encoding_format" not in call_kwargs["json"]  # Gemini doesn't use this param
    assert call_kwargs["json"]["input"] == [f"data:image/jpeg;base64,{__import__('base64').b64encode(b'fake-image-bytes').decode()}"]

  def test_png_mime_type_is_correct(self):
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get", return_value=_mock_download()):
      with patch("common.multimodal_embeddings.requests.post", return_value=_mock_proxy_response()) as mock_post:
        embedder.embed_image_url("https://example.com/logo.png")

    data_uri = mock_post.call_args.kwargs["json"]["input"][0]
    assert data_uri.startswith("data:image/png;base64,")

  def test_proxy_error_response_raises_runtime_error(self):
    embedder = self._make_embedder()
    with patch("common.multimodal_embeddings.requests.get", return_value=_mock_download()):
      with patch("common.multimodal_embeddings.requests.post", side_effect=requests.HTTPError("400 Bad Request")):
        with pytest.raises(RuntimeError, match="Embeddings proxy request failed"):
          embedder.embed_image_url("https://example.com/photo.jpg")


class TestMultimodalEmbeddingsFactory:
  def test_defaults_to_gemini(self):
    with patch.dict(os.environ, ENV, clear=True):
      embedder = MultimodalEmbeddingsFactory.get_embedder()
      assert isinstance(embedder, GeminiMultimodalEmbedder)

  def test_can_select_nova(self):
    with patch.dict(os.environ, {**ENV, "MULTIMODAL_EMBEDDINGS_PROVIDER": "nova"}):
      embedder = MultimodalEmbeddingsFactory.get_embedder()
      assert isinstance(embedder, NovaMultimodalEmbedder)

  def test_dimension_matches_selected_provider(self):
    with patch.dict(os.environ, {**ENV, "MULTIMODAL_EMBEDDINGS_PROVIDER": "nova"}):
      assert MultimodalEmbeddingsFactory.get_embedding_dimension() == NovaMultimodalEmbedder.dimension

  def test_invalid_provider_raises_value_error(self):
    with patch.dict(os.environ, {**ENV, "MULTIMODAL_EMBEDDINGS_PROVIDER": "bogus"}):
      with pytest.raises(ValueError, match="Unsupported multimodal embeddings provider"):
        MultimodalEmbeddingsFactory.get_embedder()
