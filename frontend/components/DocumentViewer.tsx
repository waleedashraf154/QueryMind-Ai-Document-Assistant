"use client";

import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import { getDocumentFileUrl, getDocumentPreview, fetchDocumentBytesForPreview } from "@/lib/api";
import PdfViewer from "@/components/PdfViewer";
import { extractPdfSections, ReaderSectionsView, type ReaderSection } from "@/components/PdfTextReader";

const DEFAULT_ZOOM = 80;

export default function DocumentViewer({
  filename,
  initialSnippet,
  onClose,
}: {
  filename: string;
  initialSnippet?: string;
  onClose: () => void;
}) {
  const [zoom, setZoom] = useState(DEFAULT_ZOOM);
  const [loading, setLoading] = useState(true);
  const [docText, setDocText] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState("");
  const [fileSize, setFileSize] = useState<number | undefined>(undefined);
  const [error, setError] = useState<string>("");
  const [pdfBlobUrl, setPdfBlobUrl] = useState<string | null>(null);
  const [pdfData, setPdfData] = useState<Uint8Array | null>(null);
  const [viewMode, setViewMode] = useState<"visual" | "text">("visual");
  const [pdfTextSections, setPdfTextSections] = useState<ReaderSection[]>([]);
  const [pdfTextLoading, setPdfTextLoading] = useState(false);
  const blobUrlRef = useRef<string | null>(null);

  const ext = (filename.slice(filename.lastIndexOf(".")).toLowerCase()) || "";
  const isPdf = ext === ".pdf";
  const isImage = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"].includes(ext);
  const isTextDoc = [".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md", ".json"].includes(ext);

  const fileUrl = getDocumentFileUrl(filename);

  // Revoke object URL on unmount
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, []);

  const handlePdfError = useCallback((msg: string) => {
    setError(msg);
  }, []);

  useEffect(() => {
    let active = true;

    setLoading(true);
    setError("");
    setZoom(DEFAULT_ZOOM);
    setSearchQuery("");
    setPdfBlobUrl(null);
    setPdfData(null);
    setViewMode("visual");
    setPdfTextSections([]);
    setPdfTextLoading(false);

    if (isPdf) {
      console.log(`[PREVIEW] click/start: '${filename}'`);
      console.log(`[PREVIEW] document fetch start: '${filename}'`);
      const tFetchStart = performance.now();

      const timeoutId = setTimeout(() => {
        if (active) {
          console.warn(`[PREVIEW] Load timed out for '${filename}'`);
          setError("Unable to load PDF preview. Please try again.");
          setLoading(false);
        }
      }, 15000);

      // Use POST request so external download managers (like IDM) NEVER intercept the preview stream
      fetchDocumentBytesForPreview(filename)
        .then((buffer) => {
          if (!active) return;
          clearTimeout(timeoutId);

          const tFetchEnd = performance.now();
          console.log(
            `[PREVIEW] document fetch end (duration=${(tFetchEnd - tFetchStart).toFixed(1)}ms, bytes=${buffer.byteLength})`
          );
          console.log(`[PREVIEW] fetch count = 1`);

          const uint8 = new Uint8Array(buffer);
          setPdfData(uint8);
          setFileSize(buffer.byteLength);

          console.log(`[PREVIEW] blob/object-url start`);
          const pdfBlob = new Blob([buffer.slice(0)], { type: "application/pdf" });
          const url = URL.createObjectURL(pdfBlob);
          if (blobUrlRef.current) {
            URL.revokeObjectURL(blobUrlRef.current);
          }
          blobUrlRef.current = url;
          setPdfBlobUrl(url);
          console.log(`[PREVIEW] blob/object-url end`);

          // Immediately show visual PDF viewer - never wait for backend OCR/indexing/RAG!
          setLoading(false);

          // Build the Text tab from the SAME PDF bytes as the visual viewer.
          // This preserves PDF text positions/columns and lets us group content
          // under the correct heading instead of relying on the backend
          // extraction string (which can flatten a resume into one block).
          setPdfTextLoading(true);
          extractPdfSections(uint8)
            .then((sections) => {
              if (!active) return;
              setPdfTextSections(sections);
            })
            .catch(async (textErr) => {
              if (!active) return;
              console.warn("[PREVIEW] Structured PDF text extraction failed; using backend fallback.", textErr);
              try {
                const fallback = await getDocumentPreview(filename);
                if (!active) return;
                setDocText(fallback.text || "");
              } catch {}
            })
            .finally(() => {
              if (active) setPdfTextLoading(false);
            });
        })
        .catch((reason) => {
          if (!active) return;
          clearTimeout(timeoutId);
          console.error("[PREVIEW] PDF preview error:", reason);
          setError(reason instanceof Error ? reason.message : "Unable to load PDF preview. Please try again.");
          setLoading(false);
        });

      // initialSnippet is intentionally not used as the primary PDF Text view.
      // The full original PDF is parsed instead.
    } else if (isTextDoc) {
      console.log(`[PREVIEW] document fetch start (text doc): '${filename}'`);
      getDocumentPreview(filename)
        .then((data) => {
          if (!active) return;
          console.log(`[PREVIEW] document fetch end (text doc): '${filename}'`);
          setDocText(data.text || initialSnippet || "No readable text found in document.");
          if (data.size) setFileSize(data.size);
        })
        .catch((err) => {
          if (!active) return;
          if (initialSnippet) {
            setDocText(initialSnippet);
          } else {
            setError(err instanceof Error ? err.message : "Unable to load document preview. Please try again.");
          }
        })
        .finally(() => {
          if (active) setLoading(false);
        });
    } else {
      setLoading(false);
    }

    return () => {
      active = false;
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [filename, isPdf, isTextDoc, fileUrl, initialSnippet]);


  function zoomIn() {
    setZoom((z) => Math.min(z + 20, 240));
  }

  function zoomOut() {
    setZoom((z) => Math.max(z - 20, 40));
  }

  function resetZoom() {
    setZoom(DEFAULT_ZOOM);
  }

  // Non-PDF text documents still use paragraph rendering. PDFs use the
  // structure-aware reader produced directly from PDF.js text positions.
  const displayParagraphs = useMemo(() => {
    return docText
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean);
  }, [docText]);

  const filteredParagraphs = useMemo(() => {
    if (!searchQuery.trim()) return displayParagraphs;
    const q = searchQuery.toLowerCase();
    return displayParagraphs.filter((p) => p.toLowerCase().includes(q));
  }, [displayParagraphs, searchQuery]);

  const filteredPdfSections = useMemo(() => {
    if (!searchQuery.trim()) return pdfTextSections;
    const q = searchQuery.toLowerCase();
    return pdfTextSections.filter((section) =>
      `${section.heading || ""}\n${section.lines.join("\n")}`.toLowerCase().includes(q)
    );
  }, [pdfTextSections, searchQuery]);

  function escapeHtml(text: string): string {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function downloadTextHtml() {
    let body = "";

    if (isPdf && pdfTextSections.length) {
      body = pdfTextSections.map((section) => {
        const heading = section.heading
          ? `<h2>${escapeHtml(section.heading)}</h2>`
          : "";
        const lines = section.lines.map((line) => {
          const bullet = /^(?:[•▪◦●○*-]|\d+[.)])\s+/.test(line);
          const text = line.replace(/^(?:[•▪◦●○*-]|\d+[.)])\s+/, "");
          return bullet ? `<li>${escapeHtml(text)}</li>` : `<p>${escapeHtml(text)}</p>`;
        });
        const content = lines.some((line) => line.startsWith("<li>"))
          ? `<ul>${lines.filter((line) => line.startsWith("<li>")).join("")}</ul>${lines.filter((line) => !line.startsWith("<li>")).join("")}`
          : lines.join("");
        return `<section>${heading}${content}</section>`;
      }).join("\n");
    } else {
      body = displayParagraphs.map((para) => `<p>${escapeHtml(para)}</p>`).join("\n");
    }

    if (!body) {
      body = `<p>No readable text was extracted from this document.</p>`;
    }

    const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${escapeHtml(filename)}</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;line-height:1.65;max-width:900px;margin:40px auto;padding:0 24px;color:#1f2937;background:#fff}
h1{font-size:26px;margin-bottom:8px}h2{font-size:19px;margin:28px 0 12px;border-bottom:1px solid #ddd;padding-bottom:7px}p{margin:7px 0}ul{margin:7px 0 14px;padding-left:26px}li{margin:5px 0}section{margin-bottom:24px}
.meta{color:#6b7280;font-size:13px;margin-bottom:28px;border-bottom:1px solid #e5e7eb;padding-bottom:14px}
</style></head><body><h1>${escapeHtml(filename)}</h1><div class="meta">QueryMind structured text export</div>${body}</body></html>`;

    const blob = new Blob(["\ufeff", html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename.replace(/\.[^.]+$/, "") + ".html";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function highlightText(text: string, query: string) {
    if (!query.trim()) return text;
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
    const parts = text.split(regex);
    return parts.map((part, idx) =>
      idx % 2 === 1 ? (
        <mark key={idx} className="rounded bg-teal-400/30 px-1 py-0.5 font-medium text-teal-200">{part}</mark>
      ) : (
        <span key={idx}>{part}</span>
      )
    );
  }

  const formatBadgeLabel = ext.replace(".", "").toUpperCase() || "DOC";

  const formatDescription = isPdf
    ? "PDF Document"
    : isImage
    ? "Image File"
    : ext === ".docx"
    ? "Microsoft Word Document"
    : ext === ".xlsx"
    ? "Spreadsheet Document"
    : ext === ".pptx"
    ? "Presentation Deck"
    : ext === ".csv"
    ? "Comma-Separated Data"
    : "Text Document";

  const canTextRead = isPdf || isImage || isTextDoc;
  const showTextReader = canTextRead && viewMode === "text";

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-[#0d1218] text-parchment shadow-2xl">
      {/* ── Top Header Toolbar ─────────────────────────────────────── */}
      <header className="flex h-[68px] shrink-0 items-center justify-between border-b border-white/10 bg-[#121820]/95 px-5 backdrop-blur-md">
        {/* Document Title & Badge */}
        <div className="flex min-w-0 items-center gap-3 pr-4">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-teal-400/25 bg-teal-500/10 text-xs font-bold uppercase tracking-wider text-teal-300 shadow-[0_0_12px_rgba(20,184,166,0.12)]">
            {formatBadgeLabel}
          </span>
          <div className="min-w-0">
            <h2
              className="truncate text-sm font-semibold text-parchment max-w-[170px] sm:max-w-[240px] md:max-w-[320px] lg:max-w-[400px]"
              title={filename}
            >
              {filename}
            </h2>
            <div className="flex items-center gap-2 text-[11px] text-parchment/45">
              <span>{formatDescription}</span>
              {fileSize ? (
                <>
                  <span>•</span>
                  <span>{(fileSize / (1024 * 1024)).toFixed(2)} MB</span>
                </>
              ) : null}
            </div>
          </div>
        </div>

        {/* Toolbar Controls */}
        <div className="flex shrink-0 items-center gap-2">
          {/* Document View Mode Switcher (Visual vs Text) */}
          {canTextRead && (
            <div className="flex items-center rounded-xl border border-white/10 bg-white/[0.04] p-0.5 text-xs text-parchment/70 shadow-sm">
              <button
                onClick={() => setViewMode("visual")}
                className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                  viewMode === "visual"
                    ? "bg-teal-500/20 text-teal-300 shadow-sm"
                    : "text-parchment/50 hover:text-parchment"
                }`}
                title={isPdf ? "View original PDF pages" : isImage ? "View original image" : "View original file"}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
                <span>Visual</span>
              </button>
              <button
                onClick={() => {
                  setViewMode("text");
                  if (!isPdf && !docText) {
                    setLoading(true);
                    getDocumentPreview(filename)
                      .then((data) => {
                        setDocText(data.text || initialSnippet || "No readable text found in document.");
                        if (data.size) setFileSize(data.size);
                      })
                      .catch((err) => {
                        setError(err instanceof Error ? err.message : "Unable to load document text.");
                      })
                      .finally(() => setLoading(false));
                  }
                }}
                className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                  viewMode === "text"
                    ? "bg-teal-500/20 text-teal-300 shadow-sm"
                    : "text-parchment/50 hover:text-parchment"
                }`}
                title="View extracted text"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="21" y1="6" x2="3" y2="6" />
                  <line x1="15" y1="12" x2="3" y2="12" />
                  <line x1="17" y1="18" x2="3" y2="18" />
                </svg>
                <span>Text</span>
              </button>
            </div>
          )}

          {/* Zoom controls pill */}
          <div className="flex items-center rounded-xl border border-white/10 bg-white/[0.04] p-0.5 text-xs text-parchment/70 shadow-sm">
            <button
              onClick={zoomOut}
              title="Zoom out"
              aria-label="Zoom out"
              className="grid h-7 w-7 place-items-center rounded-lg text-parchment/60 transition-colors hover:bg-white/10 hover:text-parchment"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M5 12h14" />
              </svg>
            </button>
            <button
              onClick={resetZoom}
              title="Reset zoom (80%)"
              className="px-2 text-xs font-medium text-parchment/80 transition-colors hover:text-teal-300"
            >
              {zoom}%
            </button>
            <button
              onClick={zoomIn}
              title="Zoom in"
              aria-label="Zoom in"
              className="grid h-7 w-7 place-items-center rounded-lg text-parchment/60 transition-colors hover:bg-white/10 hover:text-parchment"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 5v14M5 12h14" />
              </svg>
            </button>
          </div>

          {/* Download: Visual PDF = original file, Text = structured HTML export. */}
          {canTextRead && viewMode === "text" ? (
            <button
              type="button"
              disabled={loading || pdfTextLoading}
              onClick={downloadTextHtml}
              title="Download text view as HTML"
              aria-label="Download text view as HTML"
              className="grid h-8 w-8 place-items-center rounded-xl border border-white/10 bg-white/[0.04] text-parchment/60 transition-all hover:border-white/20 hover:bg-white/10 hover:text-parchment disabled:cursor-not-allowed disabled:opacity-40"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </button>
          ) : (
            <a
              href={`${fileUrl}&download=1`}
              download={filename}
              title="Download original file"
              aria-label="Download original file"
              className="grid h-8 w-8 place-items-center rounded-xl border border-white/10 bg-white/[0.04] text-parchment/60 transition-all hover:border-white/20 hover:bg-white/10 hover:text-parchment"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </a>
          )}

          {/* Open original in new tab */}
          {isPdf ? (
            pdfBlobUrl && pdfBlobUrl.startsWith("blob:") ? (
              <a
                href={pdfBlobUrl}
                target="_blank"
                rel="noopener noreferrer"
                title="Open raw file in new tab"
                aria-label="Open raw file in new tab"
                className="grid h-8 w-8 place-items-center rounded-xl border border-white/10 bg-white/[0.04] text-parchment/60 transition-all hover:border-white/20 hover:bg-white/10 hover:text-parchment"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
                </svg>
              </a>
            ) : null
          ) : (
            <a
              href={fileUrl}
              target="_blank"
              rel="noopener noreferrer"
              title="Open raw file in new tab"
              aria-label="Open raw file in new tab"
              className="grid h-8 w-8 place-items-center rounded-xl border border-white/10 bg-white/[0.04] text-parchment/60 transition-all hover:border-white/20 hover:bg-white/10 hover:text-parchment"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
              </svg>
            </a>
          )}

          {/* Close preview button */}
          <button
            onClick={onClose}
            title="Close preview"
            aria-label="Close preview"
            className="grid h-8 w-8 place-items-center rounded-xl border border-white/10 bg-white/[0.04] text-parchment/60 transition-all hover:border-red-500/30 hover:bg-red-500/15 hover:text-red-300"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      </header>

      {/* ── Document Viewer Main Canvas ────────────────────────────── */}
      <div className="relative flex-1 min-h-0 overflow-hidden bg-[#0a0e13]">
        {/* Visual document viewer */}
        {canTextRead && viewMode === "visual" && (
          <div className="relative h-full w-full overflow-hidden bg-[#0a0e13]">
            {isPdf ? (
              loading ? (
                <div className="flex h-full w-full flex-col items-center justify-center gap-3 text-sm text-parchment/45">
                  <span className="h-6 w-6 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" />
                  <span>Loading PDF document…</span>
                </div>
              ) : error ? (
                <div className="flex h-full w-full items-center justify-center p-6">
                  <div className="mx-auto max-w-md rounded-2xl border border-red-500/30 bg-red-950/20 p-6 text-center text-sm text-red-200 shadow-xl">
                    <p className="font-semibold text-base">Unable to display PDF preview</p>
                    <p className="mt-2 text-xs text-red-200/70">{error}</p>
                    <div className="mt-5 flex items-center justify-center gap-3">
                      <button
                        onClick={() => setViewMode("text")}
                        className="rounded-xl bg-teal-500/20 px-3.5 py-2 text-xs font-medium text-teal-300 transition hover:bg-teal-500/30"
                      >
                        Switch to Text Reader
                      </button>
                      <a
                        href={`${fileUrl}&download=1`}
                        download={filename}
                        className="rounded-xl bg-white/10 px-3.5 py-2 text-xs font-medium text-white transition hover:bg-white/20"
                      >
                        Download Original
                      </a>
                    </div>
                  </div>
                </div>
              ) : pdfData ? (
                <PdfViewer data={pdfData} zoom={zoom} onError={handlePdfError} />
              ) : null
            ) : isImage ? (
              <div className="preview-scroll relative flex h-full w-full items-center justify-center overflow-auto p-6">
                <div
                  style={{
                    transform: `scale(${zoom / 100})`,
                    transformOrigin: "center center",
                    transition: "transform 0.15s cubic-bezier(0.2,0,0,1)",
                  }}
                  className="flex max-h-full max-w-full items-center justify-center"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={fileUrl}
                    alt={filename}
                    className="max-h-[82vh] max-w-full rounded-2xl border border-white/10 object-contain shadow-2xl"
                  />
                </div>
              </div>
            ) : (
              <div className="flex h-full flex-col items-center justify-center p-8 text-center">
                <div className="max-w-lg border border-white/10 bg-[#131922] p-8 shadow-2xl rounded-2xl">
                  <span className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-2xl border border-teal-400/30 bg-teal-500/15 text-lg font-bold text-teal-300">
                    {formatBadgeLabel}
                  </span>
                  <h3 className="text-base font-semibold text-white">{filename}</h3>
                  <p className="mt-2 text-xs leading-relaxed text-parchment/55">
                    The original file is ready to download. Switch to the Text tab to view extracted content and download it as HTML.
                  </p>
                  <a
                    href={`${fileUrl}&download=1`}
                    download={filename}
                    className="mt-6 inline-flex items-center gap-2 rounded-xl bg-teal-500 px-4 py-2.5 text-xs font-semibold text-slate-900 shadow-md transition hover:bg-teal-400"
                  >
                    Download Original
                  </a>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Extracted Text Reader for PDF, Office, data, text, and image files */}
        {showTextReader && (
          <div className="flex h-full w-full flex-col overflow-hidden">
            {/* Search toolbar */}
            <div className="flex shrink-0 items-center justify-between gap-3 border-b border-white/10 bg-[#11171f] px-5 py-2.5">
              <div className="relative flex-1 max-w-md">
                <svg
                  className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-parchment/40"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="11" cy="11" r="8" />
                  <path d="m21 21-4.35-4.35" />
                </svg>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search in document..."
                  className="w-full rounded-xl border border-white/10 bg-white/[0.04] py-1.5 pl-9 pr-8 text-xs text-parchment placeholder-parchment/40 outline-none transition-all focus:border-teal-400/40 focus:bg-white/[0.07]"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery("")}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-parchment/40 transition-colors hover:text-parchment"
                    title="Clear search"
                  >
                    ✕
                  </button>
                )}
              </div>

              <div className="flex items-center gap-2">
                {searchQuery && (
                  <span className="rounded-lg bg-teal-500/15 px-2 py-0.5 text-[11px] font-semibold text-teal-300">
                    {isPdf ? filteredPdfSections.length : filteredParagraphs.length}{" "}
                    {(isPdf ? filteredPdfSections.length : filteredParagraphs.length) === 1 ? "match" : "matches"}
                  </span>
                )}
                <span className="text-[11px] text-parchment/40">
                  {isPdf
                    ? `${pdfTextSections.length} ${pdfTextSections.length === 1 ? "section" : "sections"}`
                    : `${displayParagraphs.length} ${displayParagraphs.length === 1 ? "section" : "sections"}`}
                </span>
              </div>
            </div>

            {/* Document Reader Body */}
            <div className="preview-scroll flex-1 overflow-y-auto p-4 sm:p-6">
              {loading ? (
                <div className="flex h-64 flex-col items-center justify-center gap-3 text-sm text-parchment/45">
                  <span className="h-6 w-6 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" />
                  <span>Loading and extracting document content…</span>
                </div>
              ) : error && !docText ? (
                <div className="mx-auto max-w-xl rounded-2xl border border-red-500/30 bg-red-900/20 p-6 text-center text-sm text-red-200 shadow-xl">
                  <p className="font-semibold">Unable to preview document text</p>
                  <p className="mt-1 text-xs text-red-200/70">{error}</p>
                  <a
                    href={fileUrl}
                    download={filename}
                    className="mt-4 inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2 text-xs font-semibold text-white transition hover:bg-white/20"
                  >
                    Download Original File
                  </a>
                </div>
              ) : (
                <div
                  style={{
                    transform: `scale(${zoom / 100})`,
                    transformOrigin: "top center",
                    transition: "transform 0.12s ease",
                  }}
                  className="mx-auto w-full max-w-[660px] space-y-5 pb-12"
                >
                  {/* Document Page Card */}
                  <article className="rounded-2xl border border-white/10 bg-[#131922] p-6 sm:p-8 shadow-2xl leading-relaxed">
                    <header className="border-b border-white/10 pb-5 mb-6">
                      <h1 className="text-xl font-bold tracking-tight text-white">{filename}</h1>
                      <p className="mt-1 text-xs text-parchment/45">
                        High-fidelity text reader • {isPdf ? pdfTextSections.length : displayParagraphs.length} sections extracted
                      </p>
                    </header>

                    <div className="font-sans text-sm text-parchment/85 leading-7">
                      {isPdf ? (
                        pdfTextLoading ? (
                          <div className="flex h-64 items-center justify-center gap-3 text-sm text-parchment/50">
                            <span className="h-6 w-6 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" />
                            <span>Building structured PDF text view…</span>
                          </div>
                        ) : pdfTextSections.length ? (
                          <ReaderSectionsView sections={filteredPdfSections} query={searchQuery} />
                        ) : (
                          <div className="py-12 text-center text-xs text-parchment/40">
                            PDF text could not be extracted. The Visual tab still shows the original document.
                          </div>
                        )
                      ) : (
                        <>
                          {filteredParagraphs.map((para, i) => (
                            <p key={i} className="mb-4 whitespace-pre-wrap">
                              {highlightText(para, searchQuery)}
                            </p>
                          ))}

                          {filteredParagraphs.length === 0 && (
                            <div className="py-12 text-center text-xs text-parchment/40">
                              {searchQuery
                                ? `No sections matching "${searchQuery}"`
                                : "Document is empty or contains no readable text."}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </article>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Fallback for other file formats */}
        {!isPdf && !isImage && !isTextDoc && (
          <div className="flex h-full flex-col items-center justify-center p-8 text-center">
            <div className="max-w-md rounded-2xl border border-white/10 bg-[#131922] p-8 shadow-2xl">
              <span className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-2xl border border-teal-400/30 bg-teal-500/15 text-lg font-bold text-teal-300">
                {formatBadgeLabel}
              </span>
              <h3 className="text-base font-semibold text-white">{filename}</h3>
              <p className="mt-2 text-xs leading-relaxed text-parchment/50">
                Text extraction is available for supported documents and images. Visual preview is available for PDF and image files.
              </p>
              <a
                href={fileUrl}
                download={filename}
                className="mt-6 inline-flex items-center gap-2 rounded-xl bg-teal-500 px-4 py-2.5 text-xs font-semibold text-slate-900 shadow-md transition hover:bg-teal-400"
              >
                Download File
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
