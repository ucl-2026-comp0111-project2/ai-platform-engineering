// assisted-by Codex Codex-sonnet-4-6

import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { AgentTimeline } from "../DynamicAgentTimeline";
import type { TimelineData } from "@/types/dynamic-agent-timeline";

jest.mock("@/components/shared/timeline", () => ({
  CollapsibleSection: ({ children }: { children: React.ReactNode }) => <section>{children}</section>,
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
  TaskList: () => null,
}));

jest.mock("@/components/dynamic-agents/AgentAvatar", () => ({
  AgentAvatar: () => <span data-testid="agent-avatar" />,
}));

jest.mock("@/components/dynamic-agents/FileTree", () => ({
  FileTree: () => null,
}));

jest.mock("../WorkflowRunCard", () => ({
  WorkflowRunCard: () => null,
}));

function renderTimeline(data: TimelineData) {
  return render(
    <AgentTimeline
      data={data}
      files={[]}
      tasks={[]}
      isLatestMessage={true}
    />,
  );
}

describe("AgentTimeline", () => {
  it("renders Knowledge Base image search results as images", () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Here are the matching images.",
      segments: [
        {
          type: "tool",
          id: "tool-1",
          data: {
            id: "call-1",
            name: "knowledge-base_search_images",
            status: "completed",
            startedAt: new Date(),
            result: JSON.stringify({
              type: "knowledge_base_image_results",
              query: "example logo",
              results: [
                {
                  rank: 1,
                  image_url: "https://example.com/logo.png",
                  source_document: "https://example.com/source",
                  alt_text: "Example logo",
                },
              ],
            }),
          },
        },
      ],
    };

    renderTimeline(data);

    expect(screen.getByRole("img", { name: "Example logo" })).toHaveAttribute(
      "src",
      "https://example.com/logo.png",
    );
    expect(screen.getByText("#1")).toBeInTheDocument();
  });

  it("preserves the standard rendering path for non-image tool results", () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Completed.",
      segments: [{
        type: "tool",
        id: "tool-standard",
        data: {
          id: "call-standard",
          name: "knowledge-base_search",
          status: "completed",
          startedAt: new Date(),
          result: "Standard document result",
        },
      }],
    };

    renderTimeline(data);

    expect(screen.getByText("Standard document result")).toBeInTheDocument();
  });

  it("links to a safe source document when an image URL is unavailable", () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Image result.",
      segments: [{
        type: "tool",
        id: "tool-source-fallback",
        data: {
          id: "call-source-fallback",
          name: "knowledge-base_search_images",
          status: "completed",
          startedAt: new Date(),
          result: JSON.stringify({
            type: "knowledge_base_image_results",
            results: [{ rank: 1, source_document: "https://example.com/source" }],
          }),
        },
      }],
    };

    renderTimeline(data);

    expect(screen.getByText("Image unavailable")).toBeInTheDocument();
    expect(screen.getByRole("link")).toHaveAttribute("href", "https://example.com/source");
  });

  it("falls back safely when an image tool returns invalid JSON", () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Image result.",
      segments: [{
        type: "tool",
        id: "tool-invalid-json",
        data: {
          id: "call-invalid-json",
          name: "knowledge-base_search_images",
          status: "completed",
          startedAt: new Date(),
          result: "{invalid-json",
        },
      }],
    };

    renderTimeline(data);

    expect(screen.getByText("{invalid-json")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("rejects unsafe image URLs", () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Image result.",
      segments: [{
        type: "tool",
        id: "tool-unsafe",
        data: {
          id: "call-unsafe",
          name: "knowledge-base_search_images",
          status: "completed",
          startedAt: new Date(),
          result: JSON.stringify({
            type: "knowledge_base_image_results",
            results: [{ rank: 1, image_url: "javascript:alert(1)", alt_text: "Unsafe" }],
          }),
        },
      }],
    };

    renderTimeline(data);

    expect(screen.queryByRole("img", { name: "Unsafe" })).not.toBeInTheDocument();
    expect(screen.getByText("Image unavailable")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Unsafe/ })).not.toBeInTheDocument();
  });

  it("rejects image URLs containing embedded credentials", () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Image result.",
      segments: [{
        type: "tool",
        id: "tool-credentials",
        data: {
          id: "call-credentials",
          name: "knowledge-base_search_images",
          status: "completed",
          startedAt: new Date(),
          result: JSON.stringify({
            type: "knowledge_base_image_results",
            results: [{
              rank: 1,
              image_url: "https://user:password@example.com/image.png",
              alt_text: "Credential URL",
            }],
          }),
        },
      }],
    };

    renderTimeline(data);

    expect(screen.queryByRole("img", { name: "Credential URL" })).not.toBeInTheDocument();
    expect(screen.getByText("Image unavailable")).toBeInTheDocument();
  });

  it("ignores malformed image result entries", () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Image result.",
      segments: [{
        type: "tool",
        id: "tool-malformed",
        data: {
          id: "call-malformed",
          name: "knowledge-base_search_images",
          status: "completed",
          startedAt: new Date(),
          result: JSON.stringify({
            type: "knowledge_base_image_results",
            results: [{ rank: "first", image_url: "https://example.com/image.png" }],
          }),
        },
      }],
    };

    renderTimeline(data);

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("No matching images found.")).toBeInTheDocument();
  });

  it("opens image results when a running tool completes", () => {
    const running: TimelineData = {
      isStreaming: true,
      hasTools: true,
      finalAnswer: "",
      segments: [{
        type: "tool",
        id: "tool-transition",
        data: {
          id: "call-transition",
          name: "knowledge-base_search_images",
          status: "running",
          startedAt: new Date(),
        },
      }],
    };
    const { rerender } = renderTimeline(running);
    const completed: TimelineData = {
      ...running,
      isStreaming: false,
      segments: [{
        type: "tool",
        id: "tool-transition",
        data: {
          id: "call-transition",
          name: "knowledge-base_search_images",
          status: "completed",
          startedAt: new Date(),
          result: JSON.stringify({
            type: "knowledge_base_image_results",
            results: [{
              rank: 1,
              image_url: "https://example.com/logo.png",
              alt_text: "Example logo",
            }],
          }),
        },
      }],
    };

    rerender(
      <AgentTimeline data={completed} files={[]} tasks={[]} isLatestMessage={true} />,
    );

    expect(screen.getByRole("img", { name: "Example logo" })).toBeInTheDocument();
  });

  it("keeps completed turns with warnings expanded until the user collapses them", async () => {
    const data: TimelineData = {
      isStreaming: false,
      hasTools: true,
      finalAnswer: "Ready now.",
      segments: [
        {
          type: "warning",
          id: "warning-1",
          message: "MCP server is starting up and not ready yet.",
        },
        {
          type: "status",
          id: "status-1",
          status: "done",
        },
      ],
    };

    renderTimeline(data);

    const toggle = screen.getByRole("button", { name: /view execution details/i });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/MCP server is starting up/i)).toBeInTheDocument();

    fireEvent.click(toggle);

    await waitFor(() => {
      expect(toggle).toHaveAttribute("aria-expanded", "false");
    });
  });
});
