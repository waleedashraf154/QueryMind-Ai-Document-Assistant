"use client";

import { useEffect, useRef, useState } from "react";
import { registerFace, getFaceStatus, deleteFace } from "@/lib/api";
import { FaceIcon } from "@/components/Icons";

// ─── tiny icons ──────────────────────────────────────────────────────────────
function XIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7" />
      <path d="M10 11v5M14 11v5" />
    </svg>
  );
}

function RefreshIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M1 4v6h6" />
      <path d="M3.51 15a9 9 0 1 0 .49-4.95L1 10" />
    </svg>
  );
}

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

// ─── types ────────────────────────────────────────────────────────────────────
type View = "menu" | "camera";
type CameraMode = "add" | "change";

interface Props {
  onClose: () => void;
}

// ─── component ────────────────────────────────────────────────────────────────
export default function FaceIdModal({ onClose }: Props) {
  const [checkStatus, setCheckStatus] = useState<"loading" | "registered" | "none">("loading");
  const [view, setView] = useState<View>("menu");
  const [cameraMode, setCameraMode] = useState<CameraMode>("add");

  // camera state
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [camStatus, setCamStatus] = useState<"loading" | "ready" | "capturing" | "done" | "error">("loading");
  const [message, setMessage] = useState("");

  // delete state
  const [deleting, setDeleting] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  // ── Load face status ────────────────────────────────────────────────────────
  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);

    fetch("/api/face-status", { cache: "no-store", signal: controller.signal })
      .then((res) => res.json())
      .then((data: { registered?: boolean }) =>
        setCheckStatus(data.registered ? "registered" : "none"),
      )
      .catch(() => setCheckStatus("none"))
      .finally(() => clearTimeout(timer));

    return () => { controller.abort(); clearTimeout(timer); };
  }, []);

  // ── Camera lifecycle ────────────────────────────────────────────────────────
  useEffect(() => {
    if (view !== "camera") return;

    let mounted = true;
    setCamStatus("loading");
    setMessage("");

    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 640 } } })
      .then((stream) => {
        if (!mounted) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCamStatus("ready");
      })
      .catch(() => {
        if (mounted) {
          setCamStatus("error");
          setMessage("Camera access was denied. Allow camera permission and try again.");
        }
      });

    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [view]);

  // ── Capture & register ──────────────────────────────────────────────────────
  async function capture() {
    if (!videoRef.current) return;
    setCamStatus("capturing");
    setMessage("");

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Race the registration against a 35-second timeout
    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("Registration timed out. Please try again.")), 35_000),
    );

    try {
      await Promise.race([
        registerFace(canvas.toDataURL("image/jpeg", 0.88), cameraMode === "change"),
        timeoutPromise,
      ]);
      setCamStatus("done");
      setMessage(
        cameraMode === "change"
          ? "Face ID updated successfully."
          : "Face ID registered successfully. You can now sign in with your face.",
      );
      setCheckStatus("registered");
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    } catch (error) {
      setCamStatus("error");
      setMessage(error instanceof Error ? error.message : "Face registration failed. Please try again.");
    }
  }

  // ── Delete face ─────────────────────────────────────────────────────────────
  async function handleDelete() {
    setDeleting(true);
    try {
      await deleteFace();
      setCheckStatus("none");
      setDeleteConfirm(false);
      setMessage("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to delete Face ID.");
    } finally {
      setDeleting(false);
    }
  }

  // ── Open camera ─────────────────────────────────────────────────────────────
  function openCamera(mode: CameraMode) {
    setCameraMode(mode);
    setView("camera");
    setCamStatus("loading");
    setMessage("");
  }

  // ── Back to menu ────────────────────────────────────────────────────────────
  function goBack() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setView("menu");
    setCamStatus("loading");
    setMessage("");
    setDeleteConfirm(false);
  }

  // ────────────────────────────────────────────────────────────────────────────
  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      role="dialog"
      aria-modal="true"
      aria-label="Face ID settings"
    >
      {/* Panel */}
      <div className="relative w-full max-w-md rounded-3xl border border-white/10 bg-[#1c1a29] text-white shadow-2xl overflow-hidden">

        {/* Header */}
        <div className="flex items-center gap-3 border-b border-white/10 px-6 py-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-500/15 text-teal-300">
            <FaceIcon className="h-5 w-5" />
          </span>
          <div className="flex-1">
            <p className="font-semibold leading-none">Face ID</p>
            <p className="mt-0.5 text-xs text-white/40">Biometric authentication</p>
          </div>
          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-white/40 hover:bg-white/10 hover:text-white transition-colors"
            aria-label="Close"
          >
            <XIcon className="h-4 w-4" />
          </button>
        </div>

        {/* ── MENU VIEW ───────────────────────────────────────────────────── */}
        {view === "menu" && (
          <div className="px-6 py-6">
            {checkStatus === "loading" ? (
              <div className="flex flex-col items-center py-10 gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" />
                <p className="text-sm text-white/40">Checking Face ID status…</p>
              </div>
            ) : checkStatus === "registered" ? (
              /* ── Face is registered ─────────────────────────────────── */
              <>
                {/* Status badge */}
                <div className="mb-6 flex items-center gap-3 rounded-2xl border border-teal-500/20 bg-teal-500/8 px-4 py-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-teal-500/20">
                    <FaceIcon className="h-4 w-4 text-teal-300" />
                  </span>
                  <div>
                    <p className="text-sm font-medium text-teal-200">Face ID is active</p>
                    <p className="text-xs text-white/40">Your biometric is registered and ready to use</p>
                  </div>
                  <span className="ml-auto h-2 w-2 rounded-full bg-teal-400 animate-pulse" />
                </div>

                {message && (
                  <p className={`mb-4 text-center text-sm rounded-xl px-3 py-2 ${
                    message.includes("success") || message.includes("updated") || message.includes("deleted")
                      ? "bg-teal-500/10 text-teal-300"
                      : "bg-red-500/10 text-red-300"
                  }`}>
                    {message}
                  </p>
                )}

                {/* Change Face */}
                <button
                  onClick={() => openCamera("change")}
                  className="group flex w-full items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-3.5 text-left transition hover:border-teal-500/40 hover:bg-teal-500/8"
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/10 text-white/60 group-hover:bg-teal-500/20 group-hover:text-teal-300 transition-colors">
                    <RefreshIcon className="h-4 w-4" />
                  </span>
                  <div className="flex-1">
                    <p className="text-sm font-medium">Change Face</p>
                    <p className="text-xs text-white/40">Register a new face to replace the current one</p>
                  </div>
                </button>

                {/* Delete Face */}
                <div className="mt-3">
                  {!deleteConfirm ? (
                    <button
                      onClick={() => setDeleteConfirm(true)}
                      className="group flex w-full items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-3.5 text-left transition hover:border-red-500/30 hover:bg-red-500/8"
                    >
                      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/10 text-white/60 group-hover:bg-red-500/20 group-hover:text-red-400 transition-colors">
                        <TrashIcon className="h-4 w-4" />
                      </span>
                      <div className="flex-1">
                        <p className="text-sm font-medium">Delete Face</p>
                        <p className="text-xs text-white/40">Remove your registered biometric data</p>
                      </div>
                    </button>
                  ) : (
                    <div className="rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-4">
                      <p className="text-sm font-medium text-red-300">Delete Face ID?</p>
                      <p className="mt-1 text-xs text-white/45">
                        This will permanently remove your biometric data. You can re-register at any time.
                      </p>
                      <div className="mt-3 flex gap-2">
                        <button
                          onClick={handleDelete}
                          disabled={deleting}
                          className="flex-1 rounded-lg bg-red-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-600 disabled:opacity-50"
                        >
                          {deleting ? "Deleting…" : "Yes, Delete"}
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(false)}
                          className="flex-1 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold transition hover:bg-white/10"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              /* ── No face registered ─────────────────────────────────── */
              <>
                {/* Status badge */}
                <div className="mb-6 flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10">
                    <FaceIcon className="h-4 w-4 text-white/40" />
                  </span>
                  <div>
                    <p className="text-sm font-medium text-white/70">No Face ID registered</p>
                    <p className="text-xs text-white/35">Add biometric sign-in for faster access</p>
                  </div>
                </div>

                {message && (
                  <p className="mb-4 text-center text-sm text-red-300 rounded-xl bg-red-500/10 px-3 py-2">{message}</p>
                )}

                {/* Add Face */}
                <button
                  onClick={() => openCamera("add")}
                  className="group flex w-full items-center gap-3 rounded-xl border border-teal-500/30 bg-teal-500/10 px-4 py-3.5 text-left transition hover:border-teal-400/50 hover:bg-teal-500/15"
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-500/20 text-teal-300 group-hover:bg-teal-500/30 transition-colors">
                    <PlusIcon className="h-4 w-4" />
                  </span>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-teal-200">Add Face ID</p>
                    <p className="text-xs text-white/40">Register your face for biometric sign-in</p>
                  </div>
                </button>
              </>
            )}
          </div>
        )}

        {/* ── CAMERA VIEW ─────────────────────────────────────────────────── */}
        {view === "camera" && (
          <div className="px-6 py-6">
            {/* Back button */}
            <button
              onClick={goBack}
              className="mb-5 flex items-center gap-1.5 text-xs text-white/40 hover:text-white transition-colors"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back
            </button>

            {/* Title */}
            <p className="mb-1 text-base font-semibold">
              {cameraMode === "change" ? "Change Face ID" : "Register Face ID"}
            </p>
            <p className="mb-5 text-xs text-white/40">
              {cameraMode === "change"
                ? "Center your face in the frame to replace your existing biometric."
                : "Center only your face in the frame. Keep good lighting for best results."}
            </p>

            {/* Camera viewport */}
            <div className="relative mx-auto aspect-square w-full max-w-[260px] overflow-hidden rounded-[48%] border border-teal-400/40 bg-[#0f1720] p-1.5 shadow-[0_0_0_10px_rgba(47,143,130,.07)]">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="h-full w-full rounded-[46%] object-cover"
              />
              {/* Inner ring */}
              <div className="pointer-events-none absolute inset-3 rounded-[43%] border border-teal-300/30" />
              {/* Scanning animation when capturing */}
              {camStatus === "capturing" && (
                <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-[48%]">
                  <div
                    className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-teal-400 to-transparent opacity-70"
                    style={{ animation: "faceScan 1.2s ease-in-out infinite" }}
                  />
                </div>
              )}
              {/* Done overlay */}
              {camStatus === "done" && (
                <div className="absolute inset-0 flex items-center justify-center rounded-[48%] bg-teal-500/20">
                  <svg viewBox="0 0 24 24" fill="none" stroke="#5eead4" strokeWidth="2.5"
                    strokeLinecap="round" strokeLinejoin="round" className="h-14 w-14">
                    <path d="M20 6 9 17l-5-5" />
                  </svg>
                </div>
              )}
            </div>

            {/* Scan animation keyframes */}
            <style>{`
              @keyframes faceScan {
                0%   { top: 10%; opacity: 0; }
                10%  { opacity: 0.7; }
                90%  { opacity: 0.7; }
                100% { top: 90%; opacity: 0; }
              }
            `}</style>

            {/* Message */}
            {message && (
              <p className={`mt-4 text-center text-sm rounded-xl px-3 py-2 ${
                camStatus === "done"
                  ? "bg-teal-500/10 text-teal-300"
                  : "bg-red-500/10 text-red-300"
              }`}>
                {message}
              </p>
            )}

            {/* CTA */}
            {camStatus !== "done" ? (
              <button
                onClick={capture}
                disabled={camStatus !== "ready"}
                className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-teal-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-teal-500 disabled:opacity-40"
              >
                {camStatus === "capturing" ? (
                  <>
                    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="40" strokeDashoffset="10" strokeLinecap="round" />
                    </svg>
                    Registering…
                  </>
                ) : (
                  <>
                    <FaceIcon className="h-4 w-4" />
                    {camStatus === "loading"
                      ? "Starting camera…"
                      : cameraMode === "change"
                      ? "Capture & Update"
                      : "Capture & Register"}
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={onClose}
                className="mt-5 w-full rounded-xl bg-teal-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-teal-500"
              >
                Done
              </button>
            )}

            {/* Processing hint — visible only while ML is running */}
            {camStatus === "capturing" && (
              <p className="mt-2 text-center text-xs text-white/30 animate-pulse">
                Analyzing face with AI model — this may take up to 15s…
              </p>
            )}
            {camStatus === "error" && (
              <button
                onClick={() => {
                  setCamStatus("loading");
                  setMessage("");
                  // re-trigger useEffect
                  setView("menu");
                  setTimeout(() => setView("camera"), 50);
                }}
                className="mt-2 w-full rounded-xl border border-white/10 px-6 py-2.5 text-sm text-white/60 transition hover:bg-white/5"
              >
                Retry
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
