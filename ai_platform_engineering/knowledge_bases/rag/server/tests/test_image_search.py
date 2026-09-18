"""Tests for image retrieval against the CAIPE image collection."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common" / "src"))
sys.path.insert(0, str(ROOT / "server" / "src"))

from server.image_search import (  # noqa: E402
  DEFAULT_TEXT_CANDIDATES,
  _entity_to_dict,
  _find_vector_dimension,
  _find_vector_field,
  _get_collection_stats,
  _get_metric_type,
  _hit_to_result,
  _provider_compatible_results,
  _query_tokens,
  _vector_relevance,
  inspect_image_collection,
  search_image,
  search_text,
)


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


def test_inspect_image_collection_reports_missing_collection():
  client = FakeMilvusClient()
  client.has_collection = lambda collection_name: False

  info = inspect_image_collection(client=client)

  assert info.exists is False
  assert info.vector_field is None


def test_inspect_image_collection_supports_object_schema_and_index_fallback():
  class SchemaField:
    name = "vector"
    type = "FLOAT_VECTOR"
    params = {"dim": "invalid"}
    description = ""
    is_primary = False

  class Schema:
    fields = [SchemaField()]

  class Stats:
    row_count = 4

  class ObjectSchemaClient(FakeMilvusClient):
    def describe_collection(self, collection_name):
      return Schema()

    def get_collection_stats(self, collection_name):
      return Stats()

    def describe_index(self, **kwargs):
      if "index_name" not in kwargs:
        raise RuntimeError("default index unavailable")
      return {"params": {"metric_type": "L2"}}

  info = inspect_image_collection(client=ObjectSchemaClient())

  assert info.vector_field == "vector"
  assert info.vector_dimension is None
  assert info.row_count == 4
  assert info.metric_type == "L2"


def test_schema_helpers_handle_legacy_and_incomplete_vector_fields():
  legacy_fields = [{"name": "dense", "type": "FLOAT"}]

  assert _find_vector_field(legacy_fields) == "dense"
  assert _find_vector_field([{"name": "text", "type": "VARCHAR"}]) is None
  assert _find_vector_dimension(legacy_fields, None) is None
  assert _find_vector_dimension([{"name": "text"}], "dense") is None


def test_entity_and_hit_helpers_support_object_sdk_responses():
  class Entity:
    text = "https://example.com/object.png"
    pk = "object-image"
    alt_text = "Object response"

  class Hit:
    entity = Entity()
    score = 0.75
    id = None

  result = _hit_to_result(Hit(), rank=2)

  assert result.image_id == "object-image"
  assert result.image_url == "https://example.com/object.png"
  assert result.score == 0.75
  assert result.rank == 2
  assert _entity_to_dict(None) == {}


def test_entity_helper_rejects_non_mapping_to_dict_value():
  class Entity:
    def to_dict(self):
      return "not-a-mapping"

  assert _entity_to_dict(Entity()) == {}


def test_collection_metadata_helpers_handle_unavailable_values():
  class Client:
    def get_collection_stats(self, collection_name):
      raise RuntimeError("stats unavailable")

    def describe_index(self, **kwargs):
      return {"index_param": {}}

  client = Client()

  assert _get_collection_stats(client, "rag_images") is None
  assert _get_metric_type(client, "rag_images") is None


def test_provider_filter_is_disabled_when_query_provider_is_unknown():
  results = [_hit_to_result({"id": "legacy", "entity": {}}, rank=1)]

  assert _provider_compatible_results(results, None) == results


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
  client = FakeMilvusClient(dim=1024)
  embedder = FakeEmbedder(vector=[0.1, 0.2, 0.3])

  with pytest.raises(RuntimeError, match="dimension"):
    search_image(image_path, client=client, embedder=embedder)


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


def test_search_text_uses_bounded_default_candidate_pool():
  client = FakeMilvusClient()

  search_text(
    "Example diagram",
    top_k=1,
    client=client,
    embedder=FakeEmbedder(),
  )

  assert client.search_calls[0]["limit"] == DEFAULT_TEXT_CANDIDATES


def test_query_tokens_normalise_common_plural_forms():
  assert _query_tokens("diagrams errors policies boxes") == {"diagram", "error", "policy", "box"}


def test_vector_relevance_handles_empty_and_equal_scores():
  assert _vector_relevance([], "COSINE") == []

  equal_results = [
    type("Result", (), {"score": 0.5})(),
    type("Result", (), {"score": 0.5})(),
  ]
  assert _vector_relevance(equal_results, "COSINE") == [1.0, 1.0]


def test_vector_relevance_orders_l2_distance_from_low_to_high():
  results = [
    type("Result", (), {"score": 0.1})(),
    type("Result", (), {"score": 0.9})(),
  ]

  assert _vector_relevance(results, "L2") == [1.0, 0.0]


def test_search_text_reranking_matches_singular_and_plural_metadata():
  hits = [
    {
      "id": "vector_first",
      "distance": 0.10,
      "entity": {"text": "https://example.com/unrelated.png", "alt_text": "Unrelated image"},
    },
    {
      "id": "metadata_match",
      "distance": 0.12,
      "entity": {"text": "https://example.com/diagram.png", "alt_text": "Architecture diagram"},
    },
  ]

  results = search_text(
    "Show diagrams",
    top_k=2,
    candidate_k=2,
    metadata_weight=1.0,
    client=FakeMilvusClient(hits=hits),
    embedder=FakeEmbedder(),
  )

  assert [result.image_id for result in results] == ["metadata_match", "vector_first"]


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
  client = FakeMilvusClient(dim=1024)
  embedder = FakeEmbedder(vector=[0.1, 0.2, 0.3])

  with pytest.raises(RuntimeError, match="dimension"):
    search_text("Example logo", client=client, embedder=embedder)


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
      "id": "example_logo",
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

  assert [result.image_id for result in results] == ["example_logo", "plain_text"]
  assert results[0].metadata_score > results[1].metadata_score
  assert results[0].rerank_score > results[1].rerank_score


def test_search_text_rejects_invalid_metadata_weight():
  client = FakeMilvusClient()
  embedder = FakeEmbedder()

  with pytest.raises(ValueError, match="metadata_weight"):
    search_text("Example logo", metadata_weight=1.1, client=client, embedder=embedder)


@pytest.mark.parametrize("search", [search_image, search_text])
def test_search_rejects_non_positive_top_k(search, tmp_path):
  query = tmp_path / "query.png" if search is search_image else "Example logo"
  client = FakeMilvusClient()
  embedder = FakeEmbedder()

  with pytest.raises(ValueError, match="top_k"):
    search(query, top_k=0, client=client, embedder=embedder)


@pytest.mark.parametrize("search", [search_image, search_text])
def test_search_rejects_excessive_top_k(search, tmp_path):
  query = tmp_path / "query.png" if search is search_image else "Example logo"
  client = FakeMilvusClient()
  embedder = FakeEmbedder()

  with pytest.raises(ValueError, match="top_k"):
    search(query, top_k=101, client=client, embedder=embedder)


def test_search_text_rejects_invalid_candidate_k():
  client = FakeMilvusClient()
  embedder = FakeEmbedder()

  with pytest.raises(ValueError, match="candidate_k"):
    search_text("Example logo", top_k=5, candidate_k=4, client=client, embedder=embedder)


def test_search_text_rejects_empty_query():
  client = FakeMilvusClient()
  embedder = FakeEmbedder()

  with pytest.raises(ValueError, match="text query"):
    search_text("  ", client=client, embedder=embedder)
