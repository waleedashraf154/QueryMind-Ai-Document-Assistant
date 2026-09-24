"use client";

import { useEffect, useMemo, useState } from "react";
import { loadPdfJsLib } from "@/lib/pdfjs";

type PdfLine = {
  text: string;
  x: number;
  y: number;
  width: number;
  fontSize: number;
  page: number;
};

export type ReaderSection = {
  heading: string | null;
  lines: string[];
  page: number;
};

type TextItem = {
  str?: string;
  transform?: number[];
  width?: number;
  height?: number;
  hasEOL?: boolean;
};

const RESUME_HEADINGS = new Set([
  "INFO",
  "ABOUT",
  "ABOUT ME",
  "LANGUAGE",
  "LANGUAGES",
  "REFERENCE",
  "REFERENCES",
  "SOFT SKILLS",
  "CONTACT",
  "CONTACT DETAILS",
  "CONTACT INFORMATION",
  "EDUCATION",
  "EXPERTISE",
  "PROFESSIONAL EXPERIENCE",
  "WORK EXPERIENCE",
  "EMPLOYMENT HISTORY",
  "PROJECTS",
  "CERTIFICATIONS",
  "CERTIFICATION",
  "AWARDS",
  "SUMMARY",
  "PROFILE",
  "OBJECTIVE",
  "HOBBIES",
  "INTERESTS",
  "ACHIEVEMENTS",
  "PUBLICATIONS",
  "VOLUNTEERING",
  "EXTRACURRICULAR",
  "QUALIFICATIONS",
  "ACADEMIC BACKGROUND",
  "DECLARATION",
]);

function cleanText(value: string): string {
  return value.replace(/\u00a0/g, " ").replace(/[ \t]+/g, " ").trim();
}

function fontSizeFromTransform(transform?: number[]): number {
  if (!transform || transform.length < 4) return 10;
  return Math.max(6, Math.sqrt((transform[0] || 0) ** 2 + (transform[1] || 0) ** 2));
}

function isNumberedHeading(text: string): boolean {
  const match = text.match(/^\s*(?:\d+(?:\.\d+)*[.)]?|[A-Z]\.)\s+(.+)$/);
  if (!match) return false;
  const rest = cleanText(match[1]);
  if (!rest || rest.length > 110) return false;
  if (/^(?:what|who|when|where|why|how|which|can|could|does|do|is|are|will|should)\b/i.test(rest)) return false;
  if (/[?]$/.test(rest)) return false;
  return true;
}

function isMarkdownHeading(text: string): boolean {
  return /^\s*#{1,6}\s+\S+/.test(text);
}

function isResumeHeading(text: string, fontSize: number, medianFontSize: number): boolean {
  const normalized = cleanText(text).replace(/[:：]+$/, "").toUpperCase();
  if (!normalized) return false;
  if (RESUME_HEADINGS.has(normalized)) return true;
  const words = normalized.split(/\s+/);
  const shortAllCaps = words.length <= 5 && /^[A-Z0-9&/()'\- ]+$/.test(normalized);
  return shortAllCaps && fontSize >= Math.max(medianFontSize * 1.12, medianFontSize + 1.0);
}

function joinItems(items: TextItem[]): string {
  const sorted = [...items].sort((a, b) => (a.transform?.[4] || 0) - (b.transform?.[4] || 0));
  let output = "";
  let previousEnd = 0;

  for (const item of sorted) {
    const text = item.str ?? "";
    if (!text) continue;
    const x = item.transform?.[4] || 0;
    const gap = x - previousEnd;
    if (output && gap > 1.5 && !/^[,.;:!?%)]/.test(text)) output += " ";
    output += text;
    previousEnd = x + (item.width || 0);
    if (item.hasEOL) output += "\n";
  }

  return cleanText(output);
}

function makeLines(items: TextItem[], page: number): PdfLine[] {
  const usable = items.filter((item) => cleanText(item.str || "") && Array.isArray(item.transform));
  usable.sort((a, b) => {
    const ay = a.transform?.[5] || 0;
    const by = b.transform?.[5] || 0;
    if (Math.abs(ay - by) > 1.5) return by - ay;
    return (a.transform?.[4] || 0) - (b.transform?.[4] || 0);
  });

  const lines: { items: TextItem[]; y: number }[] = [];
  for (const item of usable) {
    const itemY = item.transform?.[5] || 0;
    const size = fontSizeFromTransform(item.transform);
    const tolerance = Math.max(2.5, size * 0.32);
    const last = lines[lines.length - 1];

    if (last && Math.abs(last.y - itemY) <= tolerance) {
      const lastItem = last.items[last.items.length - 1];
      const lastRight = (lastItem?.transform?.[4] || 0) + (lastItem?.width || 0);
      const itemX = item.transform?.[4] || 0;
      const horizontalGap = itemX - lastRight;

      // A large jump on the same baseline is usually a second PDF column
      // rather than another word on the same line. Split it so column-aware
      // ordering can keep each resume section together.
      if (horizontalGap <= Math.max(35, size * 5)) {
        last.items.push(item);
      } else {
        lines.push({ items: [item], y: itemY });
      }
    } else {
      lines.push({ items: [item], y: itemY });
    }
  }

  return lines
    .map((line) => {
      const items = [...line.items].sort((a, b) => (a.transform?.[4] || 0) - (b.transform?.[4] || 0));
      const x = items.reduce((min, item) => Math.min(min, item.transform?.[4] || 0), Number.POSITIVE_INFINITY);
      const right = items.reduce((max, item) => Math.max(max, (item.transform?.[4] || 0) + (item.width || 0)), 0);
      const fontSize = Math.max(...items.map((item) => fontSizeFromTransform(item.transform)));
      return {
        text: joinItems(items),
        x: Number.isFinite(x) ? x : 0,
        y: line.y,
        width: Math.max(0, right - x),
        fontSize,
        page,
      };
    })
    .filter((line) => line.text);
}

function detectColumnOrder(lines: PdfLine[], pageWidth: number): PdfLine[] {
  if (lines.length < 10) return lines;

  const candidates = lines.filter((line) => line.x > 15 && line.width < pageWidth * 0.62);
  if (candidates.length < 8) return lines;

  const xs = [...new Set(candidates.map((line) => Math.round(line.x)))]
    .sort((a, b) => a - b);
  let bestGap = 0;
  let bestSplit = 0;
  for (let i = 0; i < xs.length - 1; i++) {
    const gap = xs[i + 1] - xs[i];
    if (gap > bestGap) {
      bestGap = gap;
      bestSplit = (xs[i] + xs[i + 1]) / 2;
    }
  }

  const left = candidates.filter((line) => line.x < bestSplit);
  const right = candidates.filter((line) => line.x >= bestSplit);
  const realTwoColumn = bestGap > Math.max(55, pageWidth * 0.08) && left.length >= 4 && right.length >= 4;
  if (!realTwoColumn) return lines;

  const leftTop = Math.min(...left.map((line) => line.y));
  const rightTop = Math.min(...right.map((line) => line.y));
  const firstColumnY = Math.min(leftTop, rightTop);
  const fullWidthOrHeader = lines.filter((line) =>
    (line.y > firstColumnY + 2 && line.width >= pageWidth * 0.62) || line.x < 25 && line.width >= pageWidth * 0.5
  );

  const sortTopDown = (a: PdfLine, b: PdfLine) => {
    if (Math.abs(a.y - b.y) > 2) return b.y - a.y;
    return a.x - b.x;
  };

  return [
    ...fullWidthOrHeader.sort(sortTopDown),
    ...left.sort(sortTopDown),
    ...right.sort(sortTopDown),
  ];
}

function buildSections(lines: PdfLine[]): ReaderSection[] {
  const sizes = lines.map((line) => line.fontSize).sort((a, b) => a - b);
  const median = sizes.length ? sizes[Math.floor(sizes.length / 2)] : 10;
  const sections: ReaderSection[] = [];
  let current: ReaderSection | null = null;

  for (const line of lines) {
    const text = cleanText(line.text);
    if (!text) continue;

    const markdown = isMarkdownHeading(text);
    const numbered = isNumberedHeading(text);
    const resume = isResumeHeading(text, line.fontSize, median);
    const heading = markdown ? text.replace(/^\s*#+\s*/, "").trim() : text;

    if (markdown || numbered || resume) {
      current = { heading, lines: [], page: line.page };
      sections.push(current);
      continue;
    }

    if (!current) {
      current = { heading: null, lines: [], page: line.page };
      sections.push(current);
    }

    current.lines.push(text);
  }

  return sections.filter((section) => section.heading || section.lines.length > 0);
}

export async function extractPdfSections(data: Uint8Array): Promise<ReaderSection[]> {
  const pdfjs = await loadPdfJsLib();
  const copy = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const loadingTask = pdfjs.getDocument({ data: new Uint8Array(copy) });
  const pdf = await loadingTask.promise;

  const allSections: ReaderSection[] = [];
  try {
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
      const page = await pdf.getPage(pageNumber);
      const content = await page.getTextContent({ includeMarkedContent: false });
      const items = (content.items || []) as TextItem[];
      const lines = makeLines(items, pageNumber);
      const ordered = detectColumnOrder(lines, page.view[2] || page.getViewport({ scale: 1 }).width);
      const pageSections = buildSections(ordered);

      if (pageSections.length === 0 && ordered.length) {
        allSections.push({ heading: null, lines: ordered.map((line) => line.text), page: pageNumber });
      } else {
        allSections.push(...pageSections);
      }
    }
  } finally {
    try { await pdf.destroy(); } catch {}
  }

  return allSections;
}

function isBullet(line: string): boolean {
  return /^(?:[•▪◦●○*-]|\d+[.)])\s+/.test(line);
}

function stripBullet(line: string): string {
  return line.replace(/^(?:[•▪◦●○*-]|\d+[.)])\s+/, "");
}

export function ReaderSectionsView({ sections, query }: { sections: ReaderSection[]; query: string }) {
  const lower = query.trim().toLowerCase();
  const visible = useMemo(
    () => !lower ? sections : sections.filter((section) =>
      `${section.heading || ""}\n${section.lines.join("\n")}`.toLowerCase().includes(lower)
    ),
    [sections, lower]
  );

  return (
    <div className="space-y-5 font-sans text-sm text-parchment/90 leading-7">
      {visible.map((section, index) => (
        <section key={`${section.page}-${index}`} className="py-4 sm:py-5 first:pt-0 last:pb-0">
          {section.heading ? (
            <h3 className="mb-3 border-b border-white/10 pb-2 text-[15px] font-bold tracking-tight text-white">
              {highlightSearch(section.heading, query)}
            </h3>
          ) : null}
          <div className="space-y-1.5">
            {section.lines.map((line, lineIndex) => {
              const bullet = isBullet(line);
              return bullet ? (
                <p key={lineIndex} className="pl-5 relative whitespace-pre-wrap">
                  <span className="absolute left-1 top-0.5 text-teal-300">•</span>
                  {highlightSearch(stripBullet(line), query)}
                </p>
              ) : (
                <p key={lineIndex} className="whitespace-pre-wrap">
                  {highlightSearch(line, query)}
                </p>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

function highlightSearch(text: string, query: string) {
  if (!query.trim()) return text;
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`(${escaped})`, "gi");
  const parts = text.split(regex);
  return parts.map((part, index) =>
    index % 2 === 1 ? (
      <mark key={index} className="rounded bg-teal-400/30 px-1 py-0.5 font-medium text-teal-200">{part}</mark>
    ) : (
      <span key={index}>{part}</span>
    )
  );
}

export default function PdfTextReader({ data, query }: { data: Uint8Array; query: string }) {
  const [sections, setSections] = useState<ReaderSection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");

    extractPdfSections(data)
      .then((result) => {
        if (!active) return;
        setSections(result);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to extract PDF text.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => { active = false; };
  }, [data]);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center gap-3 text-sm text-parchment/50">
        <span className="h-6 w-6 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" />
        <span>Building structured text view…</span>
      </div>
    );
  }

  if (error) {
    return <div className="p-8 text-center text-sm text-red-200">{error}</div>;
  }

  return <ReaderSectionsView sections={sections} query={query} />;
}
