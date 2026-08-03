# Live Image Embedding

This document explains the current image embedding flow in the CAIPE RAG codebase and the live query-side image search helper added for local testing.

## Existing Linda Implementation

Linda's implementation covers the corpus-side, or pre-embedding, path for webpage images:

```text
web page crawl
  -> webloader extracts <img> URLs and alt text
  -> RAG ingestion reads document metadata["images"]
  -> Nova multimodal embeddings are generated for each image URL
  -> vectors are stored in the Milvus image collection
```

The implementation uses native multimodal image embeddings through Cisco LiteLLM/Bedrock, not caption text embeddings. The image itself is not stored in Milvus. Milvus stores the embedding vector and metadata such as the image URL, alt text, source document reference, and embedding provider.

Linda's latest corpus-side update keeps both Gemini Embedding 2 and Nova as possible image embedding providers, with Gemini as the current default and Nova as fallback. These vector spaces are not interchangeable, so the live query image must be embedded with the same provider that produced the stored corpus image vectors.

## Query-Side Live Image Search

The live image search helper adds the missing query side:

```text
local query image
  -> read image bytes
  -> generate multimodal embedding using the same provider as the target corpus
  -> inspect Milvus image collection schema and vector dimension
  -> search the same image collection
  -> return top-k matching image records and smoke-test metrics
  -> optionally download matched image URLs for local inspection
```

This is intended for validating image-to-image retrieval against Linda's pre-embedded webpage image corpus.

## Architecture

```text
A. Local image ingestion/query
   local PNG/JPG/GIF/WebP file
      -> NovaMultimodalEmbedder.embed_image_path(...)
      -> LiteLLM / Bedrock Nova embedding
      -> query vector

B. Webpage image extraction
   crawled HTML page
      -> webloader extracts <img src> URLs
      -> document metadata["images"]

C. Vision-model-generated text
   not used in this flow

D. Multimodal/native image embeddings
   image bytes or URL content
      -> provider-specific image embedding model
      -> currently Nova query embedding support is implemented in this helper
      -> Gemini query embedding support is to be confirmed against Linda's latest branch

E. Storage and retrieval
   corpus image vectors
      -> Milvus collection rag_images
      -> live image query searches rag_images
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `LITELLM_API_BASE` | Cisco LiteLLM base URL, for example an OpenAI-compatible `/v1` endpoint. |
| `LITELLM_API_KEY` | API key for the LiteLLM proxy. |
| `MILVUS_URI` | Milvus endpoint used by the image search CLI. Defaults to `http://localhost:19530`. |
| `IMAGE_COLLECTION_NAME` | Milvus collection containing image embeddings. Defaults to `rag_images`. |

## Provider Compatibility

Use the same embedding provider on both sides of the search:

| Corpus image provider | Query image provider required | Current query-side status |
| --- | --- | --- |
| Nova multimodal embeddings | Nova multimodal embeddings | Implemented through `NovaMultimodalEmbedder`. |
| Gemini Embedding 2 | Gemini Embedding 2 | To be confirmed after Linda's latest Gemini branch is available locally. |

The CLI supports `--embedding-provider` so searches can avoid mixing incompatible vectors:

```bash
python scripts/search_image.py --image "<path-to-query-image.png>" --top-k 5 --embedding-provider nova
```

If Linda's populated `rag_images` collection uses Gemini by default, this helper should either be run against Nova-indexed records or extended with the matching Gemini query-image embedding path before final image-to-image comparison.

## Commands

Inspect the image collection:

```bash
python scripts/search_image.py --inspect
```

Search with a local image:

```bash
python scripts/search_image.py --image "<path-to-query-image.png>" --top-k 5 --embedding-provider nova
```

Print JSON output:

```bash
python scripts/search_image.py --image "<path-to-query-image.png>" --top-k 5 --embedding-provider nova --json
```

Download the retrieved images locally for inspection:

```bash
python scripts/search_image.py --image "<path-to-query-image.png>" --top-k 5 --embedding-provider nova --download-dir ./image_search_results
```

Check a known expected corpus image URL:

```bash
python scripts/search_image.py --image "<path-to-query-image.png>" --top-k 5 --embedding-provider nova --expected-url "https://example.com/image.png"
```

## Supported Inputs

| Input/source | Current status |
| --- | --- |
| Local PNG/JPG/JPEG/GIF/WebP query image | Supported by `embed_image_path` and `scripts/search_image.py`. |
| Webpage image URLs in crawled pages | Supported by Linda's webloader ingestion path. |
| Images inside PDFs or office documents | Not implemented in this flow. |
| SVG files | Not supported by the Nova image embedding model and rejected by format validation. |
| Raw image storage in Milvus | Not implemented; only vectors and metadata are stored. |

## Relevant Files

| File | Role |
| --- | --- |
| `common/src/common/multimodal_embeddings.py` | Nova multimodal embedding client and LangChain adapter. Now supports local image bytes/files as well as image URLs. |
| `ingestors/src/ingestors/webloader/loader/parsers/generic.py` | Extracts image URLs and alt text from crawled webpages. |
| `server/src/server/ingestion.py` | Reads document image metadata and pre-embeds corpus images into the image vector store. |
| `server/src/server/restapi.py` | Creates the `rag_images` Milvus collection/vector store for image embeddings. |
| `server/src/server/image_search.py` | Live query-side image embedding and Milvus image search. |
| `scripts/search_image.py` | Source-checkout CLI wrapper for live image search. |
| `common/tests/test_multimodal_embeddings.py` | Unit tests for multimodal embedding client behavior. |
| `server/tests/test_image_search.py` | Unit tests for collection inspection, search, and smoke metrics. |

## Limitations

- The query helper is a CLI/test hook, not a CAIPE UI upload flow.
- `--download-dir` retrieves image bytes from the matched `image_url` values; it cannot recover raw image bytes from Milvus because Milvus stores vectors and metadata only.
- The corpus side currently depends on webpage images discovered by the webloader. It does not extract embedded images from PDFs or office documents.
- Query and corpus embeddings must use the same model and vector dimension. The search helper checks collection dimension when the Milvus schema exposes it.
- Provider metadata should be used to filter searches when the collection may contain both Gemini and Nova image vectors.
- The default Milvus URI may need to be overridden depending on whether the command runs on the host or inside Docker.
- Retrieval quality depends on Linda's corpus image collection already being populated.

