"""Focused tests for datasource-aware MCP image search."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from server.image_search import ImageSearchResult
from server import tools as tools_module
from server.tools import AgentTools, _image_datasource_filter


def _make_tools() -> AgentTools:
  return AgentTools(
    redis_client=MagicMock(),
    vector_db_query_service=MagicMock(),
    metadata_storage=MagicMock(),
  )


def _result(datasource_id: str = "ds-1") -> ImageSearchResult:
  return ImageSearchResult(
    rank=1,
    score=0.9,
    image_id="image-1",
    image_url="https://example.com/image.png",
    datasource_id=datasource_id,
  )


def test_image_datasource_filter_quotes_values_safely() -> None:
  assert _image_datasource_filter(['ds-1', 'quoted"id']) == (
    '_datasource_id in ["ds-1", "quoted\\"id"]'
  )


@pytest.mark.asyncio
async def test_search_images_filters_to_requested_datasource(monkeypatch: pytest.MonkeyPatch) -> None:
  agent_tools = _make_tools()
  agent_tools._resolve_accessible_datasource_ids = AsyncMock(return_value=None)
  search = MagicMock(return_value=[_result("ds-1")])
  monkeypatch.setattr(tools_module, "search_text", search)

  response = await agent_tools.search_images("architecture", datasource_id=" ds-1 ")

  search.assert_called_once_with(
    text="architecture",
    top_k=5,
    search_filter='_datasource_id in ["ds-1"]',
  )
  assert response["datasource_id"] == "ds-1"
  assert response["results"][0]["datasource_id"] == "ds-1"


@pytest.mark.asyncio
async def test_search_images_applies_all_accessible_datasources(monkeypatch: pytest.MonkeyPatch) -> None:
  agent_tools = _make_tools()
  agent_tools._resolve_accessible_datasource_ids = AsyncMock(return_value=["ds-1", "ds-2"])
  search = MagicMock(return_value=[])
  monkeypatch.setattr(tools_module, "search_text", search)

  await agent_tools.search_images("architecture")

  search.assert_called_once_with(
    text="architecture",
    top_k=5,
    search_filter='_datasource_id in ["ds-1", "ds-2"]',
  )


@pytest.mark.asyncio
async def test_search_images_does_not_query_inaccessible_datasource(monkeypatch: pytest.MonkeyPatch) -> None:
  agent_tools = _make_tools()
  agent_tools._resolve_accessible_datasource_ids = AsyncMock(return_value=["ds-1"])
  search = MagicMock(return_value=[])
  monkeypatch.setattr(tools_module, "search_text", search)

  response = await agent_tools.search_images("architecture", datasource_id="ds-2")

  search.assert_not_called()
  assert response["results"] == []
