"""Tests for image retrieval against the CAIPE image collection."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common" / "src"))
sys.path.insert(0, str(ROOT / "server" / "src"))

from server.image_search import inspect_image_collection, search_image, search_text  # noqa: E402


class FakeEmbedder:
  def __init__(self, vector=None):
    self.vector = vector or [0.1, 0.2, 0.3]
    self.paths = []
    self.texts = []

  def embed_image_path(self, image_path):
    self.paths.append(str(image_path))
    return self.vector

  def embed_text(self, text):
    self.texts.append(text)
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
          "embedding_provider": "FakeEmbedder",
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


def test_search_image_generates_embedding_and_returns_results(tmp_path):
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
  assert results[0].embedding_provider == "FakeEmbedder"
  assert client.search_calls[0]["anns_field"] == "vector"
  assert client.search_calls[0]["data"] == [[0.1, 0.2, 0.3]]
  assert client.search_calls[0]["limit"] == 3


def test_search_image_excludes_explicitly_incompatible_provider(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")
  client = FakeMilvusClient(hits=[
    {
      "id": "wrong-provider",
      "distance": 0.1,
      "entity": {"text": "https://example.com/wrong.png", "embedding_provider": "GeminiMultimodalEmbedder"},
    },
    {
      "id": "right-provider",
      "distance": 0.2,
      "entity": {"text": "https://example.com/right.png", "embedding_provider": "NovaMultimodalEmbedder"},
    },
  ])

  results = search_image(
    image_path,
    top_k=2,
    client=client,
    embedder=FakeEmbedder(),
    embedding_provider="NovaMultimodalEmbedder",
  )

  assert [result.image_id for result in results] == ["right-provider"]


def test_search_image_preserves_custom_filter(tmp_path):
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

  assert client.search_calls[0]["filter"] == 'source_type == "web"'


def test_search_image_allows_records_without_provider_metadata(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")
  client = FakeMilvusClient(hits=[{
    "id": "current-record",
    "distance": 0.1,
    "entity": {"text": "https://example.com/current.png"},
  }])

  results = search_image(
    image_path,
    client=client,
    embedder=FakeEmbedder(),
    embedding_provider="NovaMultimodalEmbedder",
  )

  assert [result.image_id for result in results] == ["current-record"]
  assert "filter" not in client.search_calls[0]


def test_search_image_rejects_dimension_mismatch(tmp_path):
  image_path = tmp_path / "query.png"
  image_path.write_bytes(b"fake-png")

  with pytest.raises(RuntimeError, match="dimension"):
    search_image(
      image_path,
      client=FakeMilvusClient(dim=1024),
      embedder=FakeEmbedder(vector=[0.1, 0.2, 0.3]),
    )


def test_search_text_generates_embedding_and_returns_results():
  client = FakeMilvusClient()
  embedder = FakeEmbedder()

  results = search_text(
    "Example logo with a blue circle and red mark",
    top_k=1,
    client=client,
    embedder=embedder,
    candidate_k=1,
  )

  assert embedder.texts == ["Example logo with a blue circle and red mark"]
  assert len(results) == 1
  assert results[0].image_url == "https://example.com/corpus-image.png"
  assert client.search_calls[0]["data"] == [[0.1, 0.2, 0.3]]
  assert client.search_calls[0]["limit"] == 1
  assert "filter" not in client.search_calls[0]


def test_search_text_allows_records_without_provider_metadata():
  client = FakeMilvusClient(hits=[{
    "id": "current-record",
    "distance": 0.1,
    "entity": {"text": "https://example.com/current.png", "alt_text": "Example logo"},
  }])

  results = search_text(
    "Example logo",
    top_k=1,
    client=client,
    embedder=FakeEmbedder(),
    embedding_provider="NovaMultimodalEmbedder",
  )

  assert [result.image_id for result in results] == ["current-record"]
  assert "filter" not in client.search_calls[0]


def test_search_text_excludes_explicitly_incompatible_provider():
  hits = [
    {
      "id": "wrong-provider",
      "distance": 0.1,
      "entity": {
        "text": "https://example.com/wrong.png",
        "embedding_provider": "GeminiMultimodalEmbedder",
      },
    },
    {
      "id": "unlabelled",
      "distance": 0.2,
      "entity": {"text": "https://example.com/example-logo.png", "alt_text": "Example logo"},
    },
  ]

  results = search_text(
    "Example logo",
    top_k=2,
    candidate_k=2,
    client=FakeMilvusClient(hits=hits),
    embedder=FakeEmbedder(),
    embedding_provider="NovaMultimodalEmbedder",
  )

  assert [result.image_id for result in results] == ["unlabelled"]


def test_search_text_rejects_dimension_mismatch():
  with pytest.raises(RuntimeError, match="dimension"):
    search_text(
      "Example logo",
      client=FakeMilvusClient(dim=1024),
      embedder=FakeEmbedder(vector=[0.1, 0.2, 0.3]),
    )


def test_search_text_reranks_vector_candidates_with_metadata():
  hits = [
    {
      "id": "plain_text",
      "distance": 0.10,
      "entity": {
        "text": "https://example.com/inter.png",
        "source_document": "https://example.com/brand-guidelines",
        "alt_text": "Typography sample",
      },
    },
    {
      "id": "nasa_logo",
      "distance": 0.12,
      "entity": {
        "text": "https://example.com/example-insignia-logo.png",
        "source_document": "https://example.com/brand-guidelines",
        "alt_text": "Example logo insignia",
      },
    },
  ]

  results = search_text(
    "Example logo",
    top_k=2,
    candidate_k=2,
    metadata_weight=0.8,
    client=FakeMilvusClient(hits=hits),
    embedder=FakeEmbedder(),
  )

  assert [result.image_id for result in results] == ["nasa_logo", "plain_text"]
  assert results[0].metadata_score > results[1].metadata_score
  assert results[0].rerank_score > results[1].rerank_score


def test_search_text_rejects_invalid_metadata_weight():
  with pytest.raises(ValueError, match="metadata_weight"):
    search_text(
      "Example logo",
      metadata_weight=1.1,
      client=FakeMilvusClient(),
      embedder=FakeEmbedder(),
    )


@pytest.mark.parametrize("search", [search_image, search_text])
def test_search_rejects_non_positive_top_k(search, tmp_path):
  query = tmp_path / "query.png" if search is search_image else "Example logo"
  with pytest.raises(ValueError, match="top_k"):
    search(query, top_k=0, client=FakeMilvusClient(), embedder=FakeEmbedder())


@pytest.mark.parametrize("search", [search_image, search_text])
def test_search_rejects_excessive_top_k(search, tmp_path):
  query = tmp_path / "query.png" if search is search_image else "Example logo"
  with pytest.raises(ValueError, match="top_k"):
    search(query, top_k=101, client=FakeMilvusClient(), embedder=FakeEmbedder())


def test_search_text_rejects_invalid_candidate_k():
  with pytest.raises(ValueError, match="candidate_k"):
    search_text(
      "Example logo",
      top_k=5,
      candidate_k=4,
      client=FakeMilvusClient(),
      embedder=FakeEmbedder(),
    )


def test_search_text_rejects_empty_query():
  with pytest.raises(ValueError, match="text query"):
    search_text("  ", client=FakeMilvusClient(), embedder=FakeEmbedder())
