"""Tests for live image search against the CAIPE image collection."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common" / "src"))
sys.path.insert(0, str(ROOT / "server" / "src"))

from server.image_search import (  # noqa: E402
  compute_smoke_metrics,
  download_image_results,
  inspect_image_collection,
  search_image,
)


class FakeEmbedder:
  def __init__(self, vector=None):
    self.vector = vector or [0.1, 0.2, 0.3]
    self.paths = []

  def embed_image_path(self, image_path):
    self.paths.append(str(image_path))
    return self.vector


class FakeMilvusClient:
  def __init__(self, dim=3, hits=None):
    self.dim = dim
    self.hits = hits or [
      {
        "id": "img_123",
        "distance": 0.91,
        "entity": {
          "text": "https://example.com/corpus-image.png",
          "alt_text": "Example diagram",
          "source_document": "https://example.com/page",
          "embedding_provider": "nova",
        },
      }
    ]
    self.search_calls = []

  def has_collection(self, collection_name):
    return collection_name == "rag_images"

  def describe_collection(self, collection_name):
    return {
      "fields": [
        {"name": "pk", "type": "INT64"},
        {"name": "vector", "type": "FLOAT_VECTOR", "params": {"dim": self.dim}},
        {"name": "text", "type": "VARCHAR"},
        {"name": "alt_text", "type": "VARCHAR"},
        {"name": "source_document", "type": "VARCHAR"},
        {"name": "embedding_provider", "type": "VARCHAR"},
      ]
    }

  def get_collection_stats(self, collection_name):
    return {"row_count": 2}

  def describe_index(self, **kwargs):
    return {"index_param": {"metric_type": "COSINE"}}

  def search(self, **kwargs):
    self.search_calls.append(kwargs)
    return [self.hits]


def test_inspect_image_collection_reports_schema():
  info = inspect_image_collection(client=FakeMilvusClient())

  assert info.exists is True
  assert info.collection_name == "rag_images"
  assert info.row_count == 2
  assert info.vector_field == "vector"
  assert info.vector_dimension == 3
  assert info.metric_type == "COSINE"
  assert "alt_text" in info.output_fields


def test_search_image_uses_live_embedding_and_returns_results(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")
  client = FakeMilvusClient()
  embedder = FakeEmbedder()

  results = search_image(image_path, top_k=1, client=client, embedder=embedder)

  assert embedder.paths == [str(image_path)]
  assert len(results) == 1
  assert results[0].rank == 1
  assert results[0].score == 0.91
  assert results[0].image_id == "img_123"
  assert results[0].image_url == "https://example.com/corpus-image.png"
  assert results[0].embedding_provider == "nova"
  assert client.search_calls[0]["anns_field"] == "vector"
  assert client.search_calls[0]["data"] == [[0.1, 0.2, 0.3]]
  assert client.search_calls[0]["limit"] == 1


def test_search_image_filters_by_embedding_provider(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")
  client = FakeMilvusClient()

  search_image(
    image_path,
    top_k=2,
    client=client,
    embedder=FakeEmbedder(),
    embedding_provider="nova",
  )

  assert client.search_calls[0]["filter"] == '(embedding_provider == "nova")'


def test_search_image_combines_provider_and_custom_filter(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")
  client = FakeMilvusClient()

  search_image(
    image_path,
    client=client,
    embedder=FakeEmbedder(),
    embedding_provider="nova",
    search_filter='source_type == "web"',
  )

  assert client.search_calls[0]["filter"] == '(source_type == "web") and (embedding_provider == "nova")'


def test_search_image_rejects_unsafe_embedding_provider(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")

  with pytest.raises(ValueError, match="embedding_provider"):
    search_image(
      image_path,
      client=FakeMilvusClient(),
      embedder=FakeEmbedder(),
      embedding_provider='nova" or source_type == "web"',
    )


def test_search_image_rejects_dimension_mismatch(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")

  with pytest.raises(RuntimeError, match="dimension"):
    search_image(
      image_path,
      client=FakeMilvusClient(dim=1024),
      embedder=FakeEmbedder(vector=[0.1, 0.2, 0.3]),
    )


def test_compute_smoke_metrics_expected_url():
  results = search_image(
    "query.png",
    client=FakeMilvusClient(),
    embedder=FakeEmbedder(),
  )

  metrics = compute_smoke_metrics(
    results,
    expected_url="https://example.com/corpus-image.png",
  )

  assert metrics["result_count"] == 1
  assert metrics["expected_match_found"] is True
  assert metrics["expected_match_rank"] == 1
  assert metrics["expected_match_mrr"] == 1.0


class FakeDownloadResponse:
  headers = {"Content-Type": "image/png"}

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False

  def read(self):
    return b"fake-retrieved-image"


def test_download_image_results_writes_retrieved_image(tmp_path):
  results = search_image(
    "query.png",
    client=FakeMilvusClient(),
    embedder=FakeEmbedder(),
  )

  downloaded = download_image_results(
    results,
    tmp_path,
    opener=lambda request, timeout: FakeDownloadResponse(),
  )

  assert downloaded[0].download_error is None
  assert downloaded[0].downloaded_path is not None
  output_path = Path(downloaded[0].downloaded_path)
  assert output_path.exists()
  assert output_path.name == "01_img_123.png"
  assert output_path.read_bytes() == b"fake-retrieved-image"
