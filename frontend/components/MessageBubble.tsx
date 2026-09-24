"use client";

import React from "react";
import { VolumeIcon } from "@/components/Icons";

export type Citation = { id: string; label: string; snippet: string };
export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};

// ─────────────────────────────────────────────────────────────
// Minimal self-contained Markdown renderer
// Handles: bold, italic, inline-code, headings (h1-h3 → rendered
// as h3/h4/h5 so they fit chat bubble sizing), bullet lists,
// numbered lists, horizontal rules, pipe tables, fenced code
// blocks, and paragraph breaks.
// No external npm dependency required.
// ─────────────────────────────────────────────────────────────

type RenderedNode = React.ReactNode;

/** Render inline Markdown: **bold**, *italic*, `code`, and citation markers. */
function renderInline(
  text: string,
  citations: Citation[] | undefined,
  onCitationClick: (c: Citation) => void,
  key: string
): RenderedNode {
  // Split on inline tokens: **bold**, *italic*, `code`, [N]
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[\d+\])/g);
  return (
    <React.Fragment key={key}>
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part)) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        if (/^\*[^*]+\*$/.test(part)) {
          return <em key={i}>{part.slice(1, -1)}</em>;
        }
        if (/^`[^`]+`$/.test(part)) {
          return (
            <code
              key={i}
              style={{
                fontFamily: "monospace",
                background: "rgba(0,0,0,0.07)",
                borderRadius: 3,
                padding: "1px 4px",
                fontSize: "0.88em",
              }}
            >
              {part.slice(1, -1)}
            </code>
          );
        }
        const citMatch = part.match(/^\[(\d+)\]$/);
        if (citMatch && citations) {
          const cit = citations[Number(citMatch[1]) - 1];
          if (cit) {
            return (
              <button
                key={i}
                onClick={() => onCitationClick(cit)}
                className="citation-mark mx-0.5"
                title={cit.label}
              >
                {citMatch[1]}
              </button>
            );
          }
        }
        return <React.Fragment key={i}>{part}</React.Fragment>;
      })}
    </React.Fragment>
  );
}

/** Render a pipe-delimited table into an HTML <table>. */
function renderTable(lines: string[]): RenderedNode {
  // lines[0] = header row, lines[1] = separator, lines[2..] = data rows
  const parseRow = (line: string) =>
    line
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => cell.trim());

  const headers = parseRow(lines[0]);
  const bodyRows = lines.slice(2).map(parseRow);

  return (
    <div
      style={{ overflowX: "auto", margin: "8px 0" }}
    >
      <table
        style={{
          borderCollapse: "collapse",
          width: "100%",
          fontSize: "0.88em",
          lineHeight: 1.4,
        }}
      >
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th
                key={i}
                style={{
                  border: "1px solid rgba(0,0,0,0.15)",
                  padding: "5px 10px",
                  background: "rgba(0,0,0,0.05)",
                  textAlign: "left",
                  fontWeight: 600,
                  whiteSpace: "nowrap",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bodyRows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  style={{
                    border: "1px solid rgba(0,0,0,0.12)",
                    padding: "4px 10px",
                    verticalAlign: "top",
                  }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Check whether a line is a Markdown table separator (|---|---|). */
function isTableSep(line: string): boolean {
  return /^\|?[\s\-|:]+\|[\s\-|:]+\|?$/.test(line.trim());
}

/** Check whether a line looks like a pipe-table row. */
function isTableRow(line: string): boolean {
  const t = line.trim();
  return t.startsWith("|") && t.endsWith("|") && t.includes("|", 1);
}

/**
 * Parse Markdown text and return an array of React nodes.
 * Handles:
 *   - Fenced code blocks (``` ... ```)
 *   - Pipe tables (| header | header | ... with separator row)
 *   - ATX headings (# / ## / ###)
 *   - Horizontal rules (--- / ***)
 *   - Unordered lists (- / * / •)
 *   - Ordered lists (1. / 2. ...)
 *   - Blank-line-separated paragraphs
 *   - Inline bold, italic, code
 */
function parseMarkdown(
  content: string,
  citations: Citation[] | undefined,
  onCitationClick: (c: Citation) => void
): RenderedNode[] {
  const nodes: RenderedNode[] = [];
  const lines = content.split("\n");
  let i = 0;
  let keyIdx = 0;
  const k = () => String(keyIdx++);

  while (i < lines.length) {
    const line = lines[i];

    // ── Fenced code block ─────────────────────────────────────
    if (/^```/.test(line.trim())) {
      const lang = line.trim().slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      nodes.push(
        <pre
          key={k()}
          style={{
            background: "rgba(0,0,0,0.06)",
            borderRadius: 6,
            padding: "10px 14px",
            overflowX: "auto",
            fontSize: "0.85em",
            lineHeight: 1.55,
            margin: "8px 0",
            fontFamily: "monospace",
          }}
        >
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // ── Pipe table ────────────────────────────────────────────
    if (isTableRow(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const tableLines: string[] = [line];
      i++;
      // consume separator
      tableLines.push(lines[i]);
      i++;
      // consume data rows
      while (i < lines.length && isTableRow(lines[i])) {
        tableLines.push(lines[i]);
        i++;
      }
      nodes.push(<React.Fragment key={k()}>{renderTable(tableLines)}</React.Fragment>);
      continue;
    }

    // ── Heading ───────────────────────────────────────────────
    const headMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (headMatch) {
      const level = headMatch[1].length; // 1, 2, or 3
      const text = headMatch[2].trim();
      // Map # → h3, ## → h4, ### → h5 so headings fit inside chat bubble
      const Tag = (["h3", "h4", "h5"] as const)[level - 1];
      nodes.push(
        <Tag
          key={k()}
          style={{ margin: "10px 0 4px", fontWeight: 700, lineHeight: 1.3 }}
        >
          {renderInline(text, citations, onCitationClick, k())}
        </Tag>
      );
      i++;
      continue;
    }

    // ── Horizontal rule ───────────────────────────────────────
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
      nodes.push(
        <hr
          key={k()}
          style={{ border: "none", borderTop: "1px solid rgba(0,0,0,0.15)", margin: "10px 0" }}
        />
      );
      i++;
      continue;
    }

    // ── Unordered list ────────────────────────────────────────
    if (/^(\s*)([-*•])\s+/.test(line)) {
      const items: RenderedNode[] = [];
      while (i < lines.length && /^(\s*)([-*•])\s+/.test(lines[i])) {
        const text = lines[i].replace(/^(\s*)([-*•])\s+/, "");
        items.push(
          <li key={i} style={{ marginBottom: 2 }}>
            {renderInline(text, citations, onCitationClick, k())}
          </li>
        );
        i++;
      }
      nodes.push(
        <ul
          key={k()}
          style={{ paddingLeft: 20, margin: "6px 0", listStyleType: "disc" }}
        >
          {items}
        </ul>
      );
      continue;
    }

    // ── Ordered list ──────────────────────────────────────────
    if (/^\d+\.\s+/.test(line)) {
      const items: RenderedNode[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        const text = lines[i].replace(/^\d+\.\s+/, "");
        items.push(
          <li key={i} style={{ marginBottom: 2 }}>
            {renderInline(text, citations, onCitationClick, k())}
          </li>
        );
        i++;
      }
      nodes.push(
        <ol
          key={k()}
          style={{ paddingLeft: 20, margin: "6px 0", listStyleType: "decimal" }}
        >
          {items}
        </ol>
      );
      continue;
    }

    // ── Blank line — paragraph separator ─────────────────────
    if (line.trim() === "") {
      i++;
      continue;
    }

    // ── Paragraph (possibly multi-line) ──────────────────────
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^(#{1,3})\s+/.test(lines[i]) &&
      !/^(-{3,}|\*{3,}|_{3,})$/.test(lines[i].trim()) &&
      !/^(\s*)([-*•])\s+/.test(lines[i]) &&
      !/^\d+\.\s+/.test(lines[i]) &&
      !/^```/.test(lines[i].trim()) &&
      !isTableRow(lines[i])
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      const combined = paraLines.join(" ");
      nodes.push(
        <p key={k()} style={{ margin: "4px 0", lineHeight: 1.6 }}>
          {renderInline(combined, citations, onCitationClick, k())}
        </p>
      );
    }
  }

  return nodes;
}

// ─────────────────────────────────────────────────────────────
// MessageBubble component
// ─────────────────────────────────────────────────────────────

export default function MessageBubble({
  message,
  onCitationClick,
}: {
  message: Message;
  onCitationClick: (citation: Citation) => void;
}) {
  const isUser = message.role === "user";

  function speak() {
    if (typeof window !== "undefined") {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(
        new SpeechSynthesisUtterance(message.content)
      );
    }
  }

  return (
    <div className={`flex w-full min-w-0 ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`min-w-0 ${isUser ? "max-w-[85%]" : "max-w-full"} rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${
          isUser
            ? "bg-ink text-parchment"
            : "border border-slate/10 bg-white text-slate"
        }`}
      >
        {isUser ? (
          // User messages: plain text, no Markdown parsing
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          // Assistant messages: full Markdown rendering
          <div className="markdown-body">
            {parseMarkdown(message.content, message.citations, onCitationClick)}
          </div>
        )}

        {!isUser && (
          <button
            onClick={speak}
            className="mt-2 flex items-center gap-1.5 text-xs font-medium text-slate/45 transition hover:text-signal"
            aria-label="Read this answer aloud"
          >
            <VolumeIcon className="h-3.5 w-3.5" />
            Listen
          </button>
        )}
      </div>
    </div>
  );
}
