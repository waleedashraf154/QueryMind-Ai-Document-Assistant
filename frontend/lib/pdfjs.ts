/** Shared browser-only PDF.js loader used by both the visual viewer and text reader. */
export async function loadPdfJsLib(): Promise<any> {
  if (typeof window === "undefined") {
    throw new Error("PDF.js is only available in the browser.");
  }

  const win = window as any;
  if (win.pdfjsLib) {
    if (!win.pdfjsLib.GlobalWorkerOptions.workerSrc) {
      win.pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.js";
    }
    return win.pdfjsLib;
  }

  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[data-pdfjs="1"]');
    if (existing) {
      existing.addEventListener("load", () => {
        if (win.pdfjsLib) {
          win.pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.js";
          resolve(win.pdfjsLib);
        } else {
          reject(new Error("PDF.js failed to initialize after loading."));
        }
      }, { once: true });
      existing.addEventListener("error", () => reject(new Error("Failed to load PDF.js.")), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = "/pdf.min.js";
    script.async = true;
    script.setAttribute("data-pdfjs", "1");

    script.onload = () => {
      if (win.pdfjsLib) {
        win.pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.js";
        resolve(win.pdfjsLib);
      } else {
        reject(new Error("PDF.js failed to initialize."));
      }
    };
    script.onerror = () => reject(new Error("Failed to load PDF.js."));

    document.head.appendChild(script);
  });
}
