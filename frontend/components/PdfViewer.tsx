"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";

interface PdfViewerProps {
  data: Uint8Array;
  zoom: number;
  onError?: (msg: string) => void;
  onLoaded?: (numPages: number) => void;
}

import { loadPdfJsLib } from "@/lib/pdfjs";

// Single Page Canvas Renderer with Progressive Lazy Rendering
function PdfPageCanvas({
  pdfDoc,
  pageNumber,
  zoom,
  onPageRendered,
}: {
  pdfDoc: PDFDocumentProxy;
  pageNumber: number;
  zoom: number;
  onPageRendered?: (pageNumber: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const activeTaskRef = useRef<RenderTask | null>(null);
  const [pageDimensions, setPageDimensions] = useState<{ width: number; height: number } | null>(null);
  // Page 1 is always visible immediately on initial mount
  const [isVisible, setIsVisible] = useState(pageNumber === 1);
  const [rendering, setRendering] = useState(false);
  const [hasRendered, setHasRendered] = useState(false);

  // Lazy viewport intersection observer
  useEffect(() => {
    // Single page mode or Page 1 mounts visible immediately
    if (pageNumber === 1) {
      setIsVisible(true);
    }

    const el = containerRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setIsVisible(true);
          } else {
            // Cancel active render task if user scrolled past quickly
            if (activeTaskRef.current) {
              try {
                activeTaskRef.current.cancel();
              } catch {}
              activeTaskRef.current = null;
            }
            // If already rendered, we keep it; or offload if page is far
          }
        }
      },
      {
        rootMargin: "600px 0px 600px 0px", // Preload 1-2 pages ahead
        threshold: 0.01,
      }
    );

    observer.observe(el);
    return () => {
      observer.disconnect();
    };
  }, [pageNumber]);

  // Page rendering effect
  useEffect(() => {
    if (!isVisible || !pdfDoc) return;
    let isCancelled = false;

    async function renderPage() {
      if (!canvasRef.current || !pdfDoc) return;
      setRendering(true);

      const tPageStart = performance.now();
      if (pageNumber === 1) {
        console.log("[PREVIEW] first page render start");
      }

      // Cancel any ongoing render task on this canvas
      if (activeTaskRef.current) {
        try {
          activeTaskRef.current.cancel();
        } catch {}
        activeTaskRef.current = null;
      }

      try {
        const page = await pdfDoc.getPage(pageNumber);
        if (isCancelled || !canvasRef.current) return;

        // Cap DPR at 2.0 to avoid excessive memory on ultra-high-DPI monitors
        const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;
        const scaleFactor = (zoom / 100) * 1.25;
        const viewport = page.getViewport({ scale: scaleFactor * dpr });
        const cssWidth = Math.round(viewport.width / dpr);
        const cssHeight = Math.round(viewport.height / dpr);

        setPageDimensions({ width: cssWidth, height: cssHeight });

        const canvas = canvasRef.current;
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.width = `${cssWidth}px`;
        canvas.style.height = `${cssHeight}px`;

        const ctx = canvas.getContext("2d", { alpha: false });
        if (!ctx) return;

        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const renderTask = page.render({
          canvasContext: ctx,
          viewport: viewport,
        });

        activeTaskRef.current = renderTask;
        await renderTask.promise;

        if (!isCancelled) {
          setHasRendered(true);
          const tPageEnd = performance.now();
          if (pageNumber === 1) {
            console.log(`[PREVIEW] first page render end (duration=${(tPageEnd - tPageStart).toFixed(1)}ms)`);
          }
          onPageRendered?.(pageNumber);
        }
      } catch (err: unknown) {
        const name = (err as { name?: string })?.name;
        if (name !== "RenderingCancelledException") {
          console.warn(`[PdfViewer] Error rendering page ${pageNumber}:`, err);
        }
      } finally {
        if (!isCancelled) {
          setRendering(false);
          activeTaskRef.current = null;
        }
      }
    }

    void renderPage();

    return () => {
      isCancelled = true;
      if (activeTaskRef.current) {
        try {
          activeTaskRef.current.cancel();
        } catch {}
        activeTaskRef.current = null;
      }
    };
  }, [isVisible, pdfDoc, pageNumber, zoom, onPageRendered]);

  // Estimated height before page dimensions are resolved
  const estimatedWidth = Math.round(600 * (zoom / 100));
  const estimatedHeight = Math.round(estimatedWidth * 1.35);

  return (
    <div
      ref={containerRef}
      className="relative mx-auto my-4 flex flex-col items-center"
      data-page-number={pageNumber}
    >
      <div
        className="relative overflow-hidden rounded-xl bg-white shadow-2xl transition-all duration-150 ring-1 ring-black/10"
        style={{
          width: pageDimensions ? `${pageDimensions.width}px` : `${estimatedWidth}px`,
          minHeight: pageDimensions ? `${pageDimensions.height}px` : `${estimatedHeight}px`,
        }}
      >
        <canvas
          ref={canvasRef}
          className={`block ${!hasRendered && !rendering ? "invisible" : ""}`}
        />
        {(!hasRendered || rendering) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-[#121820]/40 text-parchment/50 backdrop-blur-[1px]">
            <span className="h-6 w-6 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" />
            <span className="text-[11px] font-medium tracking-wide">Page {pageNumber}</span>
          </div>
        )}
      </div>
      <span className="mt-2 text-[11px] font-medium tracking-wide text-parchment/40">
        Page {pageNumber} of {pdfDoc.numPages}
      </span>
    </div>
  );
}

export default function PdfViewer({ data, zoom, onError, onLoaded }: PdfViewerProps) {
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [continuousView, setContinuousView] = useState(true);
  const [loadingDoc, setLoadingDoc] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const renderedPagesRef = useRef<Set<number>>(new Set());

  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const onLoadedRef = useRef(onLoaded);
  onLoadedRef.current = onLoaded;

  const handlePageRendered = useCallback((pageNum: number) => {
    renderedPagesRef.current.add(pageNum);
    if (pdfDoc && renderedPagesRef.current.size === pdfDoc.numPages) {
      console.log(`[PREVIEW] total pages render end: ${renderedPagesRef.current.size}/${pdfDoc.numPages} rendered`);
    }
  }, [pdfDoc]);

  // Initialize PDF.js and load document
  useEffect(() => {
    let cancelled = false;
    let loadingTask: any = null;

    setLoadingDoc(true);
    setLoadError(null);
    renderedPagesRef.current.clear();

    async function initPdf() {
      try {
        if (!data || data.byteLength === 0) {
          throw new Error("Document buffer is empty.");
        }

        console.log("[PREVIEW] PDF parser start");
        const tParseStart = performance.now();

        const pdfjs = await loadPdfJsLib();
        if (cancelled) return;

        // PDF.js transfers the underlying ArrayBuffer to the WebWorker via postMessage transfer list,
        // which DETACHES it from the main thread.
        // Slicing an independent copy ensures StrictMode and rerenders never receive a detached buffer.
        const bufferCopy = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
        const dataCopy = new Uint8Array(bufferCopy);

        loadingTask = pdfjs.getDocument({
          data: dataCopy,
          cMapUrl: "https://unpkg.com/pdfjs-dist@3.11.174/cmaps/",
          cMapPacked: true,
        });

        const doc = await loadingTask.promise;
        if (cancelled) {
          try { doc.destroy(); } catch {}
          return;
        }

        const tParseEnd = performance.now();
        console.log(
          `[PREVIEW] PDF parser end (duration=${(tParseEnd - tParseStart).toFixed(1)}ms, numPages=${doc.numPages})`
        );

        setPdfDoc(doc);
        setNumPages(doc.numPages);
        setCurrentPage(1);
        if (onLoadedRef.current) onLoadedRef.current(doc.numPages);
      } catch (err: unknown) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : "Unable to load PDF preview. Please try again.";
        console.error("[PREVIEW] PDF parse failed:", err);
        setLoadError(msg);
        if (onErrorRef.current) onErrorRef.current(msg);
      } finally {
        if (!cancelled) {
          setLoadingDoc(false);
        }
      }
    }

    void initPdf();

    return () => {
      cancelled = true;
      if (loadingTask) {
        try {
          loadingTask.destroy();
        } catch {}
      }
    };
  }, [data]);

  const handlePrevPage = useCallback(() => {
    setCurrentPage((p) => Math.max(p - 1, 1));
  }, []);

  const handleNextPage = useCallback(() => {
    setCurrentPage((p) => Math.min(p + 1, numPages));
  }, [numPages]);

  if (loadingDoc) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-3 text-sm text-parchment/60">
        <span className="h-7 w-7 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" />
        <p className="font-medium">Rendering PDF preview…</p>
        <span className="text-xs text-parchment/40">Preparing visual document pages</span>
      </div>
    );
  }

  if (loadError || !pdfDoc) {
    return (
      <div className="flex h-full w-full items-center justify-center p-6">
        <div className="mx-auto max-w-md rounded-2xl border border-red-500/30 bg-red-950/20 p-6 text-center text-sm text-red-200 shadow-xl">
          <p className="font-semibold text-base">Unable to load PDF preview. Please try again.</p>
          <p className="mt-2 text-xs text-red-200/70">{loadError || "Document could not be parsed."}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0d1218]">
      {/* ── PDF Page Navigation Sub-toolbar ──────────────────────── */}
      <div className="flex shrink-0 items-center justify-between border-b border-white/10 bg-[#121820] px-4 py-2 text-xs">
        {/* Page navigator */}
        <div className="flex items-center gap-2">
          {!continuousView ? (
            <div className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] p-1">
              <button
                onClick={handlePrevPage}
                disabled={currentPage <= 1}
                title="Previous page"
                className="grid h-6 w-6 place-items-center rounded text-parchment/60 transition hover:bg-white/10 hover:text-parchment disabled:opacity-30"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M15 18l-6-6 6-6" />
                </svg>
              </button>
              <span className="px-2 font-medium text-parchment/80">
                {currentPage} / {numPages}
              </span>
              <button
                onClick={handleNextPage}
                disabled={currentPage >= numPages}
                title="Next page"
                className="grid h-6 w-6 place-items-center rounded text-parchment/60 transition hover:bg-white/10 hover:text-parchment disabled:opacity-30"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M9 18l6-6-6-6" />
                </svg>
              </button>
            </div>
          ) : (
            <span className="text-[11px] font-medium text-parchment/60">
              {numPages} {numPages === 1 ? "page" : "pages"} • Continuous view
            </span>
          )}
        </div>

        {/* View mode toggle: Continuous vs Single Page */}
        <div className="flex items-center gap-1 rounded-lg border border-white/10 bg-white/[0.03] p-0.5 text-[11px]">
          <button
            onClick={() => setContinuousView(true)}
            className={`rounded-md px-2.5 py-1 font-medium transition ${
              continuousView
                ? "bg-teal-500/20 text-teal-300 shadow-sm"
                : "text-parchment/50 hover:text-parchment"
            }`}
          >
            All Pages
          </button>
          <button
            onClick={() => setContinuousView(false)}
            className={`rounded-md px-2.5 py-1 font-medium transition ${
              !continuousView
                ? "bg-teal-500/20 text-teal-300 shadow-sm"
                : "text-parchment/50 hover:text-parchment"
            }`}
          >
            Single Page
          </button>
        </div>
      </div>

      {/* ── PDF Content Scrollable Canvas Area ────────────────────── */}
      <div className="preview-scroll flex-1 overflow-auto bg-[#0a0e13] p-4 sm:p-6">
        {continuousView ? (
          <div className="flex flex-col items-center">
            {Array.from({ length: numPages }, (_, i) => (
              <PdfPageCanvas
                key={i + 1}
                pdfDoc={pdfDoc}
                pageNumber={i + 1}
                zoom={zoom}
                onPageRendered={handlePageRendered}
              />
            ))}
          </div>
        ) : (
          <div className="flex min-h-full items-center justify-center">
            <PdfPageCanvas
              key={currentPage}
              pdfDoc={pdfDoc}
              pageNumber={currentPage}
              zoom={zoom}
              onPageRendered={handlePageRendered}
            />
          </div>
        )}
      </div>
    </div>
  );
}

