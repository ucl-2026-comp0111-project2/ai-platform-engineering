# Multimodal Image Retrieval

CAIPE stores webpage image embeddings and metadata in the `rag_images` Milvus
collection. Agentic Chat retrieves these images through the Knowledge Base MCP
server without introducing a separate API or streaming protocol.

## Request flow

```text
Agentic Chat
  -> Knowledge Base search_images MCP tool
  -> provider-compatible text embedding
  -> rag_images vector search
  -> metadata reranking
  -> structured MCP result
  -> Agentic Chat image gallery
```

## Storage

The ingestion pipeline stores:

- the image embedding vector;
- the image URL;
- the source document URL;
- extracted alternative text and other document metadata.

Image bytes are not stored in Milvus. The UI loads each result from its validated
HTTP or HTTPS image URL.

## Retrieval

`server.image_search.search_text` generates a text-to-image embedding with the
configured multimodal provider, retrieves a larger vector candidate set, and
reranks candidates using:

- vector similarity;
- image URL and filename tokens;
- alternative text;
- source document metadata.

`server.image_search.search_image` provides the reusable image-to-image search
operation for callers that already have a local query image. The Agentic Chat
integration currently exposes text-to-image search through MCP.

## Provider compatibility

Query and corpus vectors must use the same embedding model and dimension.

- Collection dimensions are checked before every search.
- Legacy records that contain `embedding_provider` are rejected when they
  explicitly identify a different provider.
- Records without provider metadata remain searchable. Current operations
  require the collection to be re-indexed whenever the embedding model changes.

This supports both legacy records containing provider metadata and current
records that rely on collection-level re-indexing.

## Configuration

| Variable | Purpose |
| --- | --- |
| `LITELLM_API_BASE` | Base URL of the embeddings proxy. |
| `LITELLM_API_KEY` | API key used by the embeddings proxy. |
| `MILVUS_URI` | Milvus endpoint. Defaults to `http://localhost:19530`. |
| `IMAGE_COLLECTION_NAME` | Image collection name. Defaults to `rag_images`. |
| `ENABLE_IMAGE_EMBEDDING` | Enables image ingestion and the `search_images` MCP tool. Disabled by default. |
| `MULTIMODAL_EMBEDDINGS_PROVIDER` | Optional explicit multimodal provider selection. |
| `MAX_IMAGE_DOWNLOAD_BYTES` | Maximum downloaded image size; defaults to 10 MiB. |
| `ALLOW_NON_PUBLIC_IMAGE_URLS` | Allows image downloads from internal/private addresses. Disabled by default; enable only for trusted internal datasources. |

## MCP response

The `search_images` tool returns a `knowledge_base_image_results` object with a
bounded list of results. Each result can include:

- rank and vector score;
- reranking score;
- image ID and URL;
- source document URL;
- alternative text.

The UI validates the response shape and URL scheme before rendering a gallery.

To enable the end-to-end feature, set `ENABLE_IMAGE_EMBEDDING=true` on the RAG
server and configure an embeddings proxy with access to the selected multimodal
model. Keeping the feature opt-in preserves startup compatibility for existing
deployments that do not configure multimodal credentials.

## Limitations

- Agentic Chat currently supports text-to-image queries, not image uploads.
- Retrieval depends on the source image URL remaining accessible to the browser.
- SVG images and embedded images in PDFs or office documents are not ingested.
- Missing provider metadata cannot independently prove model compatibility;
  operational re-indexing on model changes remains required.
