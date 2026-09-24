"use client";

import { useEffect, useState } from "react";

const STAGES = [
  "Reading your documents…",
  "Finding relevant passages…",
  "Composing your answer…",
  "Almost there…",
];

export default function ProcessingIndicator() {
  const [stageIndex, setStageIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const t1 = setTimeout(() => {
      setVisible(false);
      setTimeout(() => {
        setStageIndex(1);
        setVisible(true);
      }, 150);
    }, 1000);

    const t2 = setTimeout(() => {
      setVisible(false);
      setTimeout(() => {
        setStageIndex(2);
        setVisible(true);
      }, 150);
    }, 2200);

    const t3 = setTimeout(() => {
      setVisible(false);
      setTimeout(() => {
        setStageIndex(3);
        setVisible(true);
      }, 150);
    }, 3600);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  return (
    <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2 duration-200">
      <div className="flex items-center gap-3 rounded-2xl border border-slate/10 bg-white/95 px-4 py-3 shadow-sm backdrop-blur-sm">
        <div className="relative flex h-5 w-5 shrink-0 items-center justify-center">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal/20 opacity-75" />
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-signal border-t-transparent" />
        </div>
        <p
          className={`text-sm font-medium text-slate/75 transition-opacity duration-150 ${
            visible ? "opacity-100" : "opacity-0"
          }`}
        >
          {STAGES[stageIndex]}
        </p>
      </div>
    </div>
  );
}
