"""Image-to-image and text-to-image retrieval over CAIPE image embeddings."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from common.multimodal_embeddings import BaseMultimodalEmbedder, MultimodalEmbeddingsFactory

DEFAULT_IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION_NAME", "rag_images")
DEFAULT_MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MAX_TOP_K = 100
MAX_CANDIDATES = 300
DEFAULT_OUTPUT_FIELDS = [
  "text",
  "pk",
  "source_document",
  "alt_text",
  "embedding_provider",
]
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUERY_STOP_WORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
_QUERY_SYNONYMS = {
  "logo": {"emblem", "identifier", "insignia", "logotype"},
  "emblem": {"identifier", "insignia", "logo", "logotype"},
  "insignia": {"emblem", "identifier", "logo", "logotype"},
  "logotype": {"emblem", "identifier", "insignia", "logo"},
}


@dataclass
class ImageCollectionInfo:
  collection_name: str
  exists: bool
  row_count: Optional[int] = None
  vector_field: Optional[str] = None
  vector_dimension: Optional[int] = None
  metric_type: Optional[str] = None
  output_fields: List[str] = field(default_factory=list)


@dataclass
class ImageSearchResult:
  rank: int
  score: float
  image_id: Optional[str] = None
  image_url: Optional[str] = None
  source_document: Optional[str] = None
  alt_text: Optional[str] = None
  embedding_provider: Optional[str] = None
  rerank_score: Optional[float] = None
  metadata_score: Optional[float] = None
  metadata: Dict[str, Any] = field(default_factory=dict)


def _import_milvus_client() -> Any:
  try:
    from pymilvus import MilvusClient
  except ImportError as exc:
    raise RuntimeError(
      "pymilvus is required for live image search. Install the RAG server "
      "dependencies before running this command."
    ) from exc
  return MilvusClient


def _field_dict(field: Any) -> Dict[str, Any]:
  if isinstance(field, dict):
    return field
  result: Dict[str, Any] = {}
  for name in ("name", "type", "params", "description", "is_primary"):
    if hasattr(field, name):
      result[name] = getattr(field, name)
  return result


def _schema_fields(schema: Any) -> List[Dict[str, Any]]:
  if isinstance(schema, dict):
    fields = schema.get("fields") or schema.get("schema", {}).get("fields") or []
  else:
    fields = getattr(schema, "fields", [])
  return [_field_dict(field) for field in fields]


def _field_type_name(field_type: Any) -> str:
  return str(getattr(field_type, "name", field_type)).upper()


def _find_vector_field(fields: Sequence[Dict[str, Any]]) -> Optional[str]:
  for schema_field in fields:
    field_type = _field_type_name(schema_field.get("type"))
    if "VECTOR" in field_type:
      return schema_field.get("name")
  for candidate in ("vector", "dense", "embedding"):
    if any(schema_field.get("name") == candidate for schema_field in fields):
      return candidate
  return None


def _find_vector_dimension(fields: Sequence[Dict[str, Any]], vector_field: Optional[str]) -> Optional[int]:
  if not vector_field:
    return None
  for schema_field in fields:
    if schema_field.get("name") != vector_field:
      continue
    params = schema_field.get("params") or {}
    dim = params.get("dim") or params.get("dimension")
    try:
      return int(dim) if dim is not None else None
    except (TypeError, ValueError):
      return None
  return None


def _entity_to_dict(entity: Any) -> Dict[str, Any]:
  if entity is None:
    return {}
  if isinstance(entity, dict):
    return dict(entity)
  if hasattr(entity, "to_dict"):
    value = entity.to_dict()
    return value if isinstance(value, dict) else {}
  result: Dict[str, Any] = {}
  for key in ("text", "pk", "id", "source", "source_document", "alt_text", "embedding_provider", "metadata"):
    if hasattr(entity, key):
      result[key] = getattr(entity, key)
  return result


def _hit_to_result(hit: Any, rank: int) -> ImageSearchResult:
  if isinstance(hit, dict) or (hasattr(hit, "get") and hasattr(hit, "keys")):
    entity = _entity_to_dict(hit.get("entity"))
    score = hit.get("distance", hit.get("score", 0.0))
    image_id = hit.get("id", hit.get("pk", entity.get("pk") or entity.get("id")))
  else:
    entity = _entity_to_dict(getattr(hit, "entity", None))
    score = getattr(hit, "distance", getattr(hit, "score", 0.0))
    image_id = getattr(hit, "id", None) or entity.get("pk") or entity.get("id")

  metadata = dict(entity)
  nested_metadata = entity.get("metadata") if isinstance(entity.get("metadata"), dict) else {}
  image_url = (
    entity.get("image_url")
    or entity.get("url")
    or entity.get("text")
    or entity.get("page_content")
  )
  embedding_provider = entity.get("embedding_provider") or nested_metadata.get("embedding_provider")

  return ImageSearchResult(
    rank=rank,
    score=float(score),
    image_id=str(image_id) if image_id is not None else None,
    image_url=str(image_url) if image_url is not None else None,
    source_document=entity.get("source_document") or entity.get("source"),
    alt_text=entity.get("alt_text"),
    embedding_provider=str(embedding_provider) if embedding_provider is not None else None,
    metadata=metadata,
  )


def _embedder_provider_id(embedder: BaseMultimodalEmbedder) -> str:
  """Return the provider identifier used by legacy image records."""
  return embedder.__class__.__name__


def _provider_compatible_results(
  results: Sequence[ImageSearchResult],
  embedding_provider: Optional[str],
) -> List[ImageSearchResult]:
  """Exclude only records that explicitly identify a different provider.

  Current image records rely on mandatory re-indexing when the configured
  embedding model changes and may not include provider metadata. Older records
  can include ``embedding_provider``; those records are checked when present.
  """
  if not embedding_provider:
    return list(results)
  return [
    result
    for result in results
    if result.embedding_provider in (None, embedding_provider)
  ]


def _validate_search_limits(top_k: int, candidate_k: Optional[int] = None) -> None:
  if not 1 <= top_k <= MAX_TOP_K:
    raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
  if candidate_k is not None and not top_k <= candidate_k <= MAX_CANDIDATES:
    raise ValueError(
      f"candidate_k must be between top_k and {MAX_CANDIDATES}"
    )


def _get_collection_stats(client: Any, collection_name: str) -> Optional[int]:
  try:
    stats = client.get_collection_stats(collection_name)
  except Exception:
    return None
  if isinstance(stats, dict):
    row_count = stats.get("row_count") or stats.get("num_entities")
  else:
    row_count = getattr(stats, "row_count", None)
  try:
    return int(row_count) if row_count is not None else None
  except (TypeError, ValueError):
    return None


def _get_metric_type(client: Any, collection_name: str) -> Optional[str]:
  for kwargs in ({"collection_name": collection_name}, {"collection_name": collection_name, "index_name": ""}):
    try:
      index_info = client.describe_index(**kwargs)
    except Exception:
      continue
    if isinstance(index_info, dict):
      params = index_info.get("index_param") or index_info.get("params") or index_info
      metric = params.get("metric_type") if isinstance(params, dict) else None
      if metric:
        return str(metric)
  return None


def inspect_image_collection(
  milvus_uri: str = DEFAULT_MILVUS_URI,
  collection_name: str = DEFAULT_IMAGE_COLLECTION,
  client: Optional[Any] = None,
) -> ImageCollectionInfo:
  """Inspect the image collection that stores pre-embedded corpus images."""
  if client is None:
    client = _import_milvus_client()(uri=milvus_uri)

  exists = bool(client.has_collection(collection_name))
  if not exists:
    return ImageCollectionInfo(collection_name=collection_name, exists=False)

  schema = client.describe_collection(collection_name)
  fields = _schema_fields(schema)
  field_names = [field.get("name") for field in fields if field.get("name")]
  vector_field = _find_vector_field(fields)

  return ImageCollectionInfo(
    collection_name=collection_name,
    exists=True,
    row_count=_get_collection_stats(client, collection_name),
    vector_field=vector_field,
    vector_dimension=_find_vector_dimension(fields, vector_field),
    metric_type=_get_metric_type(client, collection_name),
    output_fields=field_names,
  )


def search_image(
  image_path: str | Path,
  top_k: int = 5,
  milvus_uri: str = DEFAULT_MILVUS_URI,
  collection_name: str = DEFAULT_IMAGE_COLLECTION,
  output_fields: Optional[Sequence[str]] = None,
  embedder: Optional[BaseMultimodalEmbedder] = None,
  client: Optional[Any] = None,
  embedding_provider: Optional[str] = None,
  search_filter: Optional[str] = None,
) -> List[ImageSearchResult]:
  """Embed a local image and search the pre-embedded image collection."""
  _validate_search_limits(top_k)
  if client is None:
    client = _import_milvus_client()(uri=milvus_uri)
  info = inspect_image_collection(milvus_uri, collection_name, client=client)
  if not info.exists:
    raise RuntimeError(f"Milvus collection does not exist: {collection_name}")
  if not info.vector_field:
    raise RuntimeError(f"Could not determine vector field for collection: {collection_name}")

  embedder = embedder or MultimodalEmbeddingsFactory.get_embedder()
  embedding_provider = embedding_provider or _embedder_provider_id(embedder)
  query_embedding = embedder.embed_image_path(image_path)
  if info.vector_dimension and info.vector_dimension != len(query_embedding):
    raise RuntimeError(
      f"Query embedding dimension {len(query_embedding)} does not match "
      f"collection dimension {info.vector_dimension}. Check that corpus and "
      "query embeddings use the same model."
    )

  retrieval_limit = top_k * 3
  search_kwargs = {
    "collection_name": collection_name,
    "data": [query_embedding],
    "anns_field": info.vector_field,
    "limit": retrieval_limit,
    "output_fields": list(output_fields or DEFAULT_OUTPUT_FIELDS),
  }
  if search_filter:
    search_kwargs["filter"] = search_filter

  raw_results = client.search(**search_kwargs)
  hits: Iterable[Any] = raw_results[0] if raw_results else []
  candidates = [_hit_to_result(hit, rank=index + 1) for index, hit in enumerate(hits)]
  compatible = _provider_compatible_results(candidates, embedding_provider)[:top_k]
  for rank, result in enumerate(compatible, 1):
    result.rank = rank
  return compatible


def search_text(
  text: str,
  top_k: int = 5,
  milvus_uri: str = DEFAULT_MILVUS_URI,
  collection_name: str = DEFAULT_IMAGE_COLLECTION,
  output_fields: Optional[Sequence[str]] = None,
  embedder: Optional[BaseMultimodalEmbedder] = None,
  client: Optional[Any] = None,
  embedding_provider: Optional[str] = None,
  search_filter: Optional[str] = None,
  candidate_k: Optional[int] = None,
  metadata_weight: float = 0.55,
) -> List[ImageSearchResult]:
  """Embed text, retrieve image candidates, and rerank with image metadata."""
  _validate_search_limits(top_k, candidate_k)
  if not text.strip():
    raise ValueError("text query must not be empty")
  if not 0.0 <= metadata_weight <= 1.0:
    raise ValueError("metadata_weight must be between 0 and 1")
  if client is None:
    client = _import_milvus_client()(uri=milvus_uri)
  info = inspect_image_collection(milvus_uri, collection_name, client=client)
  if not info.exists:
    raise RuntimeError(f"Milvus collection does not exist: {collection_name}")
  if not info.vector_field:
    raise RuntimeError(f"Could not determine vector field for collection: {collection_name}")

  embedder = embedder or MultimodalEmbeddingsFactory.get_embedder()
  embedding_provider = embedding_provider or _embedder_provider_id(embedder)
  query_embedding = embedder.embed_text(text)
  if info.vector_dimension and info.vector_dimension != len(query_embedding):
    raise RuntimeError(
      f"Query embedding dimension {len(query_embedding)} does not match "
      f"collection dimension {info.vector_dimension}. Check that corpus and "
      "query embeddings use the same model."
    )

  retrieval_limit = max(top_k, candidate_k or top_k * 3)
  search_kwargs = {
    "collection_name": collection_name,
    "data": [query_embedding],
    "anns_field": info.vector_field,
    "limit": retrieval_limit,
    "output_fields": list(output_fields or DEFAULT_OUTPUT_FIELDS),
  }
  if search_filter:
    search_kwargs["filter"] = search_filter

  raw_results = client.search(**search_kwargs)
  hits: Iterable[Any] = raw_results[0] if raw_results else []
  candidates = [_hit_to_result(hit, rank=index + 1) for index, hit in enumerate(hits)]
  candidates = _provider_compatible_results(candidates, embedding_provider)
  return _rerank_text_results(
    text,
    candidates,
    top_k=top_k,
    metric_type=info.metric_type,
    metadata_weight=metadata_weight,
  )


def _query_tokens(text: str) -> set[str]:
  tokens = {token for token in _TOKEN_RE.findall(text.lower()) if token not in _QUERY_STOP_WORDS}
  expanded = set(tokens)
  for token in tokens:
    expanded.update(_QUERY_SYNONYMS.get(token, set()))
  return expanded


def _token_coverage(query_tokens: set[str], value: Optional[str]) -> float:
  if not query_tokens or not value:
    return 0.0
  value_tokens = set(_TOKEN_RE.findall(value.lower()))
  # Metadata fields are short. Cap the denominator so one or two strong
  # filename/alt-text matches are not diluted by a long descriptive query.
  return min(1.0, len(query_tokens & value_tokens) / min(4, len(query_tokens)))


def _metadata_relevance(query_tokens: set[str], result: ImageSearchResult) -> float:
  image_url = result.image_url or ""
  filename = Path(urlparse(image_url).path).name
  return (
    0.25 * _token_coverage(query_tokens, image_url)
    + 0.30 * _token_coverage(query_tokens, filename)
    + 0.30 * _token_coverage(query_tokens, result.alt_text)
    + 0.15 * _token_coverage(query_tokens, result.source_document)
  )


def _vector_relevance(results: Sequence[ImageSearchResult], metric_type: Optional[str]) -> List[float]:
  if not results:
    return []
  scores = [result.score for result in results]
  low, high = min(scores), max(scores)
  if high == low:
    return [1.0] * len(results)
  if (metric_type or "").upper() in {"COSINE", "IP"}:
    return [(score - low) / (high - low) for score in scores]
  return [(high - score) / (high - low) for score in scores]


def _rerank_text_results(
  text: str,
  results: Sequence[ImageSearchResult],
  top_k: int,
  metric_type: Optional[str],
  metadata_weight: float,
) -> List[ImageSearchResult]:
  query_tokens = _query_tokens(text)
  vector_scores = _vector_relevance(results, metric_type)
  for result, vector_score in zip(results, vector_scores):
    result.metadata_score = _metadata_relevance(query_tokens, result)
    result.rerank_score = (
      (1.0 - metadata_weight) * vector_score
      + metadata_weight * result.metadata_score
    )
  reranked = sorted(results, key=lambda result: result.rerank_score or 0.0, reverse=True)[:top_k]
  for rank, result in enumerate(reranked, 1):
    result.rank = rank
  return reranked

