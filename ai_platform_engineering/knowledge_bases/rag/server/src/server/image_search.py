"""Live image search against the CAIPE image vector collection.

This module implements the query side for image-to-image and text-to-image retrieval. Linda's
multimodal ingestion stores webpage image embeddings in Milvus. This module
embeds a local query image or text query with the configured multimodal model and
searches the same image collection.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from common.multimodal_embeddings import BaseMultimodalEmbedder, MultimodalEmbeddingsFactory

DEFAULT_IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION_NAME", "rag_images")
DEFAULT_MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DEFAULT_OUTPUT_FIELDS = [
  "text",
  "pk",
  "source_document",
  "alt_text",
  "embedding_provider",
]
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_FILTER_VALUE_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
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
  downloaded_path: Optional[str] = None
  download_error: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)


def _import_milvus_client():
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
  for field in fields:
    field_type = _field_type_name(field.get("type"))
    if "VECTOR" in field_type:
      return field.get("name")
  for candidate in ("vector", "dense", "embedding"):
    if any(field.get("name") == candidate for field in fields):
      return candidate
  return None


def _find_vector_dimension(fields: Sequence[Dict[str, Any]], vector_field: Optional[str]) -> Optional[int]:
  if not vector_field:
    return None
  for field in fields:
    if field.get("name") != vector_field:
      continue
    params = field.get("params") or {}
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


def _provider_filter(embedding_provider: Optional[str]) -> Optional[str]:
  if not embedding_provider:
    return None
  if not _SAFE_FILTER_VALUE_RE.match(embedding_provider):
    raise ValueError(
      "embedding_provider may only contain letters, numbers, '.', '_', ':', '/', or '-'."
    )
  return f'embedding_provider == "{embedding_provider}"'


def _embedder_provider_id(embedder: BaseMultimodalEmbedder) -> str:
  """Return the provider identifier persisted by image ingestion."""
  return embedder.__class__.__name__


def _combine_filters(*filters: Optional[str]) -> Optional[str]:
  active = [expr for expr in filters if expr]
  if not active:
    return None
  return " and ".join(f"({expr})" for expr in active)

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

  milvus_filter = _combine_filters(search_filter, _provider_filter(embedding_provider))
  search_kwargs = {
    "collection_name": collection_name,
    "data": [query_embedding],
    "anns_field": info.vector_field,
    "limit": top_k,
    "output_fields": list(output_fields or DEFAULT_OUTPUT_FIELDS),
  }
  if milvus_filter:
    search_kwargs["filter"] = milvus_filter

  raw_results = client.search(**search_kwargs)
  hits: Iterable[Any] = raw_results[0] if raw_results else []
  return [_hit_to_result(hit, rank=index + 1) for index, hit in enumerate(hits)]


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

  milvus_filter = _combine_filters(search_filter, _provider_filter(embedding_provider))
  retrieval_limit = max(top_k, candidate_k or top_k * 3)
  search_kwargs = {
    "collection_name": collection_name,
    "data": [query_embedding],
    "anns_field": info.vector_field,
    "limit": retrieval_limit,
    "output_fields": list(output_fields or DEFAULT_OUTPUT_FIELDS),
  }
  if milvus_filter:
    search_kwargs["filter"] = milvus_filter

  raw_results = client.search(**search_kwargs)
  hits: Iterable[Any] = raw_results[0] if raw_results else []
  candidates = [_hit_to_result(hit, rank=index + 1) for index, hit in enumerate(hits)]
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


def _safe_filename(value: Optional[str], fallback: str) -> str:
  candidate = value or fallback
  cleaned = _SAFE_FILENAME_RE.sub("_", candidate).strip("._")
  return cleaned[:120] or fallback


def _image_extension(image_url: str, content_type: Optional[str]) -> str:
  suffix = Path(urlparse(image_url).path).suffix.lower()
  if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
    return suffix
  if content_type:
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    if guessed:
      return guessed
  return ".img"


def download_image_result(
  result: ImageSearchResult,
  download_dir: str | Path,
  timeout: int = 20,
  opener=None,
) -> ImageSearchResult:
  """Download the retrieved image URL so the match can be inspected locally."""
  if not result.image_url:
    result.download_error = "No image_url available on this result."
    return result

  output_dir = Path(download_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  stem = _safe_filename(result.image_id or Path(urlparse(result.image_url).path).stem, f"image_{result.rank}")

  try:
    request = Request(result.image_url, headers={"User-Agent": "CAIPE-image-search/1.0"})
    open_fn = opener or urlopen
    with open_fn(request, timeout=timeout) as response:
      payload = response.read()
      content_type = response.headers.get("Content-Type") if hasattr(response, "headers") else None
    extension = _image_extension(result.image_url, content_type)
    output_path = output_dir / f"{result.rank:02d}_{stem}{extension}"
    output_path.write_bytes(payload)
    result.downloaded_path = str(output_path)
    result.download_error = None
  except (OSError, URLError, ValueError) as exc:
    result.download_error = str(exc)
  return result


def download_image_results(
  results: Sequence[ImageSearchResult],
  download_dir: str | Path,
  timeout: int = 20,
  opener=None,
) -> List[ImageSearchResult]:
  return [download_image_result(result, download_dir, timeout=timeout, opener=opener) for result in results]

def compute_smoke_metrics(
  results: Sequence[ImageSearchResult],
  expected_id: Optional[str] = None,
  expected_url: Optional[str] = None,
) -> Dict[str, Any]:
  """Compute simple smoke-test metrics for a known target image."""
  metrics: Dict[str, Any] = {
    "result_count": len(results),
    "top_score": results[0].score if results else None,
  }
  if expected_id or expected_url:
    match_rank = None
    for result in results:
      id_matches = expected_id and result.image_id == expected_id
      url_matches = expected_url and result.image_url == expected_url
      if id_matches or url_matches:
        match_rank = result.rank
        break
    metrics["expected_match_found"] = match_rank is not None
    metrics["expected_match_rank"] = match_rank
    metrics["expected_match_mrr"] = 1.0 / match_rank if match_rank else 0.0
  return metrics


def _print_collection_info(info: ImageCollectionInfo) -> None:
  print("Image collection:")
  for key, value in asdict(info).items():
    print(f"  {key}: {value}")


def _print_results(results: Sequence[ImageSearchResult]) -> None:
  if not results:
    print("No image results returned.")
    return
  print("Image search results:")
  for result in results:
    print(f"  #{result.rank} score={result.score:.4f} id={result.image_id}")
    print(f"     image_url: {result.image_url}")
    if result.source_document:
      print(f"     source_document: {result.source_document}")
    if result.alt_text:
      print(f"     alt_text: {result.alt_text}")
    if result.embedding_provider:
      print(f"     embedding_provider: {result.embedding_provider}")
    if result.rerank_score is not None:
      print(f"     rerank_score: {result.rerank_score:.4f}")
    if result.metadata_score is not None:
      print(f"     metadata_score: {result.metadata_score:.4f}")
    if result.downloaded_path:
      print(f"     downloaded_path: {result.downloaded_path}")
    if result.download_error:
      print(f"     download_error: {result.download_error}")


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Search CAIPE pre-embedded images with an image or text query")
  parser.add_argument("--inspect", action="store_true", help="Only inspect the image collection")
  query_group = parser.add_mutually_exclusive_group()
  query_group.add_argument("--image", help="Local PNG/JPG/GIF/WebP image to embed and search with")
  query_group.add_argument("--text", help="Text description to embed and use for image retrieval")
  parser.add_argument("--top-k", type=int, default=5, help="Number of image results to return")
  parser.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI, help="Milvus URI")
  parser.add_argument("--collection-name", default=DEFAULT_IMAGE_COLLECTION, help="Image collection name")
  parser.add_argument("--expected-id", help="Optional expected image id for smoke-test metrics")
  parser.add_argument("--expected-url", help="Optional expected image URL for smoke-test metrics")
  parser.add_argument("--embedding-provider", help="Only search corpus vectors with this embedding_provider metadata value")
  parser.add_argument("--filter", help="Optional Milvus filter expression to combine with provider filtering")
  parser.add_argument("--candidate-k", type=int, help="Candidate count for text-query metadata reranking")
  parser.add_argument("--metadata-weight", type=float, default=0.55, help="Metadata weight for text-query reranking")
  parser.add_argument("--download-dir", help="Optional directory for downloaded top-k image results")
  parser.add_argument("--download-timeout", type=int, default=20, help="Seconds to wait for each image download")
  parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  if not args.inspect and not args.image and not args.text:
    parser.error("Provide --inspect, --image, or --text")

  info = inspect_image_collection(args.milvus_uri, args.collection_name)
  if args.inspect and not args.image and not args.text:
    if args.json:
      print(json.dumps({"collection": asdict(info)}, indent=2))
    else:
      _print_collection_info(info)
    return 0

  search_kwargs = {
    "top_k": args.top_k,
    "milvus_uri": args.milvus_uri,
    "collection_name": args.collection_name,
    "embedding_provider": args.embedding_provider,
    "search_filter": args.filter,
  }
  if args.text is not None:
    results = search_text(
      text=args.text,
      candidate_k=args.candidate_k,
      metadata_weight=args.metadata_weight,
      **search_kwargs,
    )
  else:
    results = search_image(image_path=args.image, **search_kwargs)
  if args.download_dir:
    results = download_image_results(
      results,
      args.download_dir,
      timeout=args.download_timeout,
    )
  metrics = compute_smoke_metrics(results, args.expected_id, args.expected_url)
  if args.json:
    print(
      json.dumps(
        {
          "collection": asdict(info),
          "results": [asdict(result) for result in results],
          "metrics": metrics,
        },
        indent=2,
      )
    )
  else:
    _print_collection_info(info)
    _print_results(results)
    print("Smoke metrics:")
    for key, value in metrics.items():
      print(f"  {key}: {value}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())




