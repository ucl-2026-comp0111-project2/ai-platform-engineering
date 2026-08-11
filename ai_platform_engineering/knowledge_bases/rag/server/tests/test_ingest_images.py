"""Tests for DocumentProcessor._ingest_images (multimodal pre-embedding ingestion)."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.documents import Document

from server.ingestion import DocumentProcessor, MAX_IMAGES_PER_DOCUMENT
from common.multimodal_embeddings import UnsupportedImageFormatError


def _make_processor(image_vstore=None) -> DocumentProcessor:
    return DocumentProcessor(
        vstore=MagicMock(),
        job_manager=AsyncMock(),
        graph_rag_enabled=False,
        image_vstore=image_vstore,
    )


def _make_image_vstore(existing_ids=None):
    """A mock image_vstore with aclient.query/aadd_embeddings wired for testing."""
    vstore = MagicMock()
    vstore.collection_name = "rag_images"
    vstore.embeddings = MagicMock()
    vstore.embeddings.embedder = MagicMock()
    vstore.aclient = AsyncMock()
    vstore.aclient.query.return_value = [{"pk": pk} for pk in (existing_ids or [])]
    vstore.aadd_embeddings = AsyncMock()
    return vstore


def _make_doc(source: str, images) -> Document:
    """Build a Document matching the real shape produced by the webloader."""
    if isinstance(images, list):
        images = json.dumps(images)
    nested = {"source": source}
    if images is not None:
        nested["images"] = images
    return Document(page_content="some page text", metadata={"metadata": nested})


def _image_id(url: str) -> str:
    return f"img_{hashlib.md5(url.encode()).hexdigest()[:12]}"


@pytest.mark.asyncio
class TestIngestImages:
    async def test_no_images_key_is_noop(self):
        image_vstore = _make_image_vstore()
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc("https://example.com/page", images=None)

        await processor._ingest_images(documents=[doc], job_id="job-1")

        image_vstore.aadd_embeddings.assert_not_awaited()

    async def test_empty_images_list_is_noop(self):
        image_vstore = _make_image_vstore()
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc("https://example.com/page", images=[])

        await processor._ingest_images(documents=[doc], job_id="job-1")

        image_vstore.aadd_embeddings.assert_not_awaited()

    async def test_new_image_is_embedded_and_stored_with_deterministic_id(self):
        image_vstore = _make_image_vstore()
        image_vstore.embeddings.embedder.embed_image_url = MagicMock(return_value=[0.1, 0.2, 0.3])
        processor = _make_processor(image_vstore=image_vstore)
        image_url = "https://example.com/ant.jpg"
        doc = _make_doc("https://example.com/page", images=[{"url": image_url, "alt_text": "An ant"}])

        await processor._ingest_images(documents=[doc], job_id="job-1")

        image_vstore.aadd_embeddings.assert_awaited_once()
        call_kwargs = image_vstore.aadd_embeddings.call_args.kwargs
        assert call_kwargs["ids"] == [_image_id(image_url)]
        assert call_kwargs["texts"] == [image_url]
        assert call_kwargs["embeddings"] == [[0.1, 0.2, 0.3]]
        assert call_kwargs["metadatas"][0]["alt_text"] == "An ant"
        assert call_kwargs["metadatas"][0]["source_document"] == "https://example.com/page"

    async def test_already_stored_image_is_skipped(self):
        image_url = "https://example.com/ant.jpg"
        existing_id = _image_id(image_url)
        image_vstore = _make_image_vstore(existing_ids=[existing_id])
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc("https://example.com/page", images=[{"url": image_url, "alt_text": "An ant"}])

        await processor._ingest_images(documents=[doc], job_id="job-1")

        image_vstore.aadd_embeddings.assert_not_awaited()

    async def test_mixed_new_and_existing_images_only_embeds_new_ones(self):
        new_url = "https://example.com/new.jpg"
        existing_url = "https://example.com/existing.jpg"
        image_vstore = _make_image_vstore(existing_ids=[_image_id(existing_url)])
        image_vstore.embeddings.embedder.embed_image_url = MagicMock(return_value=[0.1, 0.2])
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc(
            "https://example.com/page",
            images=[
                {"url": existing_url, "alt_text": ""},
                {"url": new_url, "alt_text": ""},
            ],
        )

        await processor._ingest_images(documents=[doc], job_id="job-1")

        image_vstore.aadd_embeddings.assert_awaited_once()
        call_kwargs = image_vstore.aadd_embeddings.call_args.kwargs
        assert call_kwargs["ids"] == [_image_id(new_url)]

    async def test_images_beyond_cap_are_not_embedded(self):
        image_vstore = _make_image_vstore()
        image_vstore.embeddings.embedder.embed_image_url = MagicMock(return_value=[0.1])
        processor = _make_processor(image_vstore=image_vstore)
        many_images = [{"url": f"https://example.com/img{i}.jpg", "alt_text": ""} for i in range(MAX_IMAGES_PER_DOCUMENT + 10)]
        doc = _make_doc("https://example.com/page", images=many_images)

        await processor._ingest_images(documents=[doc], job_id="job-1")

        call_kwargs = image_vstore.aadd_embeddings.call_args.kwargs
        assert len(call_kwargs["ids"]) == MAX_IMAGES_PER_DOCUMENT

    async def test_images_missing_url_are_skipped(self):
        image_vstore = _make_image_vstore()
        image_vstore.embeddings.embedder.embed_image_url = MagicMock(return_value=[0.1])
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc(
            "https://example.com/page",
            images=[{"alt_text": "no url here"}, {"url": "https://example.com/valid.jpg", "alt_text": ""}],
        )

        await processor._ingest_images(documents=[doc], job_id="job-1")

        call_kwargs = image_vstore.aadd_embeddings.call_args.kwargs
        assert len(call_kwargs["ids"]) == 1

    async def test_unsupported_format_is_skipped_without_retry(self):
        image_vstore = _make_image_vstore()
        embed_mock = MagicMock(side_effect=UnsupportedImageFormatError("bad format"))
        image_vstore.embeddings.embedder.embed_image_url = embed_mock
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc("https://example.com/page", images=[{"url": "https://example.com/pic.svg", "alt_text": ""}])

        await processor._ingest_images(documents=[doc], job_id="job-1")

        assert embed_mock.call_count == 1
        image_vstore.aadd_embeddings.assert_not_awaited()

    async def test_transient_failure_retries_then_succeeds(self):
        image_vstore = _make_image_vstore()
        embed_mock = MagicMock(side_effect=[RuntimeError("timeout"), [0.1, 0.2]])
        image_vstore.embeddings.embedder.embed_image_url = embed_mock
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc("https://example.com/page", images=[{"url": "https://example.com/pic.jpg", "alt_text": ""}])

        with patch("server.ingestion.time.sleep"):
            await processor._ingest_images(documents=[doc], job_id="job-1")

        assert embed_mock.call_count == 2
        image_vstore.aadd_embeddings.assert_awaited_once()

    async def test_all_attempts_failing_skips_image(self):
        image_vstore = _make_image_vstore()
        embed_mock = MagicMock(side_effect=RuntimeError("persistent failure"))
        image_vstore.embeddings.embedder.embed_image_url = embed_mock
        processor = _make_processor(image_vstore=image_vstore)
        doc = _make_doc("https://example.com/page", images=[{"url": "https://example.com/pic.jpg", "alt_text": ""}])

        with patch("server.ingestion.time.sleep"):
            await processor._ingest_images(documents=[doc], job_id="job-1")

        image_vstore.aadd_embeddings.assert_not_awaited()

    async def test_image_vstore_none_is_never_called(self):
        processor = _make_processor(image_vstore=None)
        assert processor.image_vstore is None



