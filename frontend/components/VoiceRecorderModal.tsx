"use client";

import { useEffect, useRef, useState, useCallback } from "react";

/* ─── Types ─────────────────────────────────────────────────── */
type Phase = "idle" | "recording" | "transcribing" | "stopped" | "error";

interface Props {
  onSend: (text: string) => void;
  onClose: () => void;
}

/* ─── Helper: format seconds ─────────────────────────────────── */
function fmt(s: number) {
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = (s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

export default function VoiceRecorderModal({ onSend, onClose }: Props) {
  const [phase, setPhase]           = useState<Phase>("idle");
  const [bars, setBars]             = useState<number[]>(Array(24).fill(4));
  const [elapsed, setElapsed]       = useState(0);
  const [transcript, setTranscript] = useState("");
  const [errMsg, setErrMsg]         = useState("");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioCtxRef      = useRef<AudioContext | null>(null);
  const sourceRef        = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef      = useRef<AnalyserNode | null>(null);
  const rafRef           = useRef<number>(0);
  const timerRef         = useRef<ReturnType<typeof setInterval> | null>(null);
  const recognRef        = useRef<any>(null);
  const streamRef        = useRef<MediaStream | null>(null);
  const chunksRef        = useRef<Blob[]>([]);
  const audioBlobRef     = useRef<Blob | null>(null);
  const sessionIdRef     = useRef<number>(0);
  const startTimeRef     = useRef<number>(0);
  const finalTextRef     = useRef<string>("");
  const isRecordingRef   = useRef<boolean>(false);
  const isCancelledRef   = useRef<boolean>(false);

  /* ── Hard cleanup of all audio devices and streams ─────────── */
  const cleanupAudio = useCallback(() => {
    isRecordingRef.current = false;
    isCancelledRef.current = true;

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }

    if (recognRef.current) {
      try {
        recognRef.current.onresult = null;
        recognRef.current.onerror = null;
        recognRef.current.onend = null;
        recognRef.current.abort();
      } catch {}
      recognRef.current = null;
    }

    if (mediaRecorderRef.current) {
      try {
        mediaRecorderRef.current.ondataavailable = null;
        mediaRecorderRef.current.onstop = null;
        if (mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
      } catch {}
      mediaRecorderRef.current = null;
    }

    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {}
      sourceRef.current = null;
    }

    if (audioCtxRef.current) {
      try {
        if (audioCtxRef.current.state !== "closed") {
          void audioCtxRef.current.close();
        }
      } catch {}
      audioCtxRef.current = null;
    }

    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
          track.enabled = false;
        });
      } catch {}
      streamRef.current = null;
    }
  }, []);

  /* ── Safely close modal and ensure microphone is released ──── */
  const handleClose = useCallback(() => {
    cleanupAudio();
    onClose();
  }, [cleanupAudio, onClose]);

  /* ── Transcribe audio blob via backend Whisper API ─────────── */
  const transcribeAudioBlob = async (blob: Blob): Promise<string> => {
    try {
      const formData = new FormData();
      formData.append("file", blob, "voice_message.webm");

      const res = await fetch("/api/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Status ${res.status}`);
      }

      const data = await res.json();
      return (data.text || "").trim();
    } catch (err) {
      console.warn("[VoiceRecorderModal] Backend transcription fallback:", err);
      return "";
    }
  };

  /* ── Start a fresh recording session (used initially and on Retry) ── */
  const startRecordingSession = useCallback(async () => {
    // 1. Stop/cleanup any previous recorder, streams, and audio devices
    cleanupAudio();
    const sessionId = ++sessionIdRef.current;
    isCancelledRef.current = false;

    // 2. Clear old audio blob, transcript, elapsed time, and errors
    setErrMsg("");
    setTranscript("");
    setElapsed(0);
    setPhase("idle");
    finalTextRef.current = "";
    chunksRef.current = [];
    audioBlobRef.current = null;
    setBars(Array(24).fill(4));

    // 3. Request fresh microphone stream
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      if (sessionIdRef.current !== sessionId || isCancelledRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
    } catch (err) {
      if (sessionIdRef.current === sessionId && !isCancelledRef.current) {
        setErrMsg("Microphone access denied. Please allow microphone permission in your browser.");
        setPhase("error");
      }
      return;
    }

    isRecordingRef.current = true;

    // 4. Waveform analyser
    try {
      const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new AudioCtxClass();
      if (ctx.state === "suspended") {
        void ctx.resume();
      }
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 64;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;

      const dataArr = new Uint8Array(analyser.frequencyBinCount);
      const BAR_COUNT = 24;

      const tick = () => {
        if (!isRecordingRef.current || sessionIdRef.current !== sessionId) return;
        analyser.getByteFrequencyData(dataArr);
        const step = Math.max(1, Math.floor(dataArr.length / BAR_COUNT));
        const next = Array.from({ length: BAR_COUNT }, (_, i) => {
          const v = dataArr[i * step] ?? 0;
          return Math.max(4, Math.round((v / 255) * 48));
        });
        setBars(next);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (audioErr) {
      console.warn("AudioContext setup warning:", audioErr);
    }

    // 5. MediaRecorder with fresh event listeners
    try {
      let mimeType = "audio/webm";
      if (typeof MediaRecorder !== "undefined") {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/webm")) {
          mimeType = "audio/webm";
        } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
          mimeType = "audio/mp4";
        }

        const recorder = new MediaRecorder(stream, { mimeType });
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
          if (sessionIdRef.current === sessionId && e.data && e.data.size > 0) {
            chunksRef.current.push(e.data);
          }
        };

        recorder.start(250);
      }
    } catch (recErr) {
      console.warn("MediaRecorder warning:", recErr);
    }

    // 6. Web Speech API for real-time live transcript
    const SpeechClass = (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;
    if (SpeechClass) {
      try {
        const recogn = new SpeechClass();
        recogn.continuous = true;
        recogn.interimResults = true;
        recogn.lang = "en-US";
        recogn.maxAlternatives = 1;

        recogn.onresult = (e: any) => {
          if (sessionIdRef.current !== sessionId || !isRecordingRef.current) return;
          let interim = "";
          for (let i = e.resultIndex; i < e.results.length; i++) {
            const res = e.results[i];
            const text = res[0]?.transcript ?? "";
            if (res.isFinal) {
              const cleaned = text.trim();
              if (cleaned) {
                finalTextRef.current = (finalTextRef.current ? finalTextRef.current + " " : "") + cleaned;
              }
            } else {
              interim = (interim ? interim + " " : "") + text.trim();
            }
          }
          const full = (finalTextRef.current + (interim ? " " + interim : "")).trim();
          if (full) {
            setTranscript(full);
          }
        };

        recogn.onerror = (e: any) => {
          if (e.error !== "no-speech") {
            console.warn("[SpeechRecognition] error:", e.error);
          }
        };

        recogn.onend = () => {
          if (isRecordingRef.current && sessionIdRef.current === sessionId && recognRef.current) {
            try {
              recognRef.current.start();
            } catch {}
          }
        };

        recogn.start();
        recognRef.current = recogn;
      } catch (speechErr) {
        console.warn("SpeechRecognition start error:", speechErr);
      }
    }

    // 7. Precise Wall-Clock Elapsed Timer
    startTimeRef.current = Date.now();
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      if (sessionIdRef.current !== sessionId) return;
      const sec = Math.max(0, Math.floor((Date.now() - startTimeRef.current) / 1000));
      setElapsed(sec);
    }, 250);

    // 8. Update UI state to recording
    setPhase("recording");
  }, [cleanupAudio]);

  /* ── Stop recording ────────────────────────────────────────── */
  const stopRecording = useCallback(async () => {
    if (!isRecordingRef.current) return;
    isRecordingRef.current = false;
    const sessionAtStop = sessionIdRef.current;

    // Freeze timer at exact wall-clock elapsed time
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    const finalElapsed = Math.max(0, Math.floor((Date.now() - startTimeRef.current) / 1000));
    setElapsed(finalElapsed);

    // Stop waveform animation
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    setBars(Array(24).fill(4));

    // Stop speech recognition
    if (recognRef.current) {
      try {
        recognRef.current.onend = null;
        recognRef.current.stop();
      } catch {}
    }

    // Stop media recorder and assemble audio blob
    let recordedBlob: Blob | null = null;
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recordedBlob = await new Promise<Blob | null>((resolve) => {
        recorder.onstop = () => {
          try {
            const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
            resolve(blob);
          } catch {
            resolve(null);
          }
        };
        try {
          recorder.stop();
        } catch {
          resolve(null);
        }
      });
    } else if (chunksRef.current.length > 0) {
      recordedBlob = new Blob(chunksRef.current, { type: "audio/webm" });
    }

    if (sessionIdRef.current !== sessionAtStop) return;

    audioBlobRef.current = recordedBlob;
    mediaRecorderRef.current = null;

    // Release all microphone handles immediately
    if (sourceRef.current) {
      try { sourceRef.current.disconnect(); } catch {}
      sourceRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      try { void audioCtxRef.current.close(); } catch {}
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
          track.enabled = false;
        } catch {}
      });
      streamRef.current = null;
    }
    if (recognRef.current) {
      try { recognRef.current.abort(); } catch {}
      recognRef.current = null;
    }

    // Check transcript: If client SpeechRecognition missed it, use backend Whisper!
    const currentText = (finalTextRef.current || transcript).trim();
    if (currentText) {
      setTranscript(currentText);
      setPhase("stopped");
    } else if (recordedBlob && recordedBlob.size > 1000) {
      setPhase("transcribing");
      const serverText = await transcribeAudioBlob(recordedBlob);
      if (sessionIdRef.current !== sessionAtStop) return;
      if (serverText) {
        setTranscript(serverText);
        finalTextRef.current = serverText;
      }
      setPhase("stopped");
    } else {
      setPhase("stopped");
    }
  }, [transcript]);

  /* ── Cleanup on unmount ────────────────────────────────────── */
  useEffect(() => {
    return () => {
      cleanupAudio();
    };
  }, [cleanupAudio]);

  /* ── Keyboard ESC ──────────────────────────────────────────── */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleClose]);

  /* ── Send transcript ───────────────────────────────────────── */
  function handleSend() {
    const text = transcript.trim();
    if (!text) return;
    cleanupAudio();
    onSend(text);
    onClose();
  }

  /* ── Dedicated Retry Action ────────────────────────────────── */
  const handleRetry = useCallback(() => {
    void startRecordingSession();
  }, [startRecordingSession]);

  /* ── Auto-start on mount ───────────────────────────────────── */
  useEffect(() => {
    void startRecordingSession();
  }, [startRecordingSession]);



  const canSend = (phase === "stopped" || phase === "idle") && transcript.trim().length > 0;

  return (
    <>
      <style>{`
        .vr-overlay {
          position: fixed; inset: 0; z-index: 9998;
          background: rgba(0,0,0,0.6);
          backdrop-filter: blur(8px);
          display: flex; align-items: center; justify-content: center;
          animation: vrFadeIn 0.18s ease;
        }
        @keyframes vrFadeIn { from { opacity:0; } to { opacity:1; } }
        .vr-modal {
          background: #12111c;
          border: 1px solid rgba(255,255,255,0.09);
          border-radius: 24px;
          padding: 28px 26px 24px;
          width: min(440px, calc(100vw - 32px));
          box-shadow: 0 32px 80px rgba(0,0,0,0.75);
          animation: vrSlideUp 0.22s cubic-bezier(0.34,1.56,0.64,1);
          display: flex;
          flex-direction: column;
          gap: 18px;
        }
        @keyframes vrSlideUp {
          from { opacity:0; transform:translateY(18px) scale(.97); }
          to   { opacity:1; transform:translateY(0) scale(1); }
        }
        .vr-header {
          display: flex; align-items: center; justify-content: space-between;
        }
        .vr-title {
          font-size: 16px; font-weight: 700;
          background: linear-gradient(135deg,#6ee7b7,#3b82f6);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          display: flex; align-items: center; gap: 8px;
        }
        .vr-close {
          width:32px; height:32px; border-radius:50%; border:none;
          background: rgba(255,255,255,0.07); color:rgba(255,255,255,0.5);
          display:flex; align-items:center; justify-content:center;
          cursor:pointer; transition:background 0.15s,color 0.15s;
          font-size:16px; line-height:1;
        }
        .vr-close:hover { background:rgba(255,255,255,0.14); color:#fff; }
        .vr-waveform {
          height: 64px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 3px;
          background: rgba(255,255,255,0.03);
          border-radius: 16px;
          padding: 0 16px;
        }
        .vr-bar {
          width: 4px;
          border-radius: 2px;
          background: linear-gradient(180deg, #6ee7b7, #3b82f6);
          transition: height 0.08s ease;
        }
        .vr-bar.idle {
          background: rgba(255,255,255,0.15);
        }
        .vr-status {
          display: flex; align-items: center; justify-content: center; gap: 10px;
          font-size: 13px; color: rgba(255,255,255,0.5);
        }
        .vr-dot {
          width:8px; height:8px; border-radius:50%;
          background:#ef4444;
          animation: vrPulse 1s ease-in-out infinite;
        }
        @keyframes vrPulse {
          0%,100% { opacity:1; transform:scale(1); }
          50%      { opacity:0.4; transform:scale(0.7); }
        }
        .vr-spinner {
          width: 14px; height: 14px;
          border: 2px solid rgba(255,255,255,0.2);
          border-top-color: #38bdf8;
          border-radius: 50%;
          animation: vrSpin 0.7s linear infinite;
        }
        @keyframes vrSpin { to { transform: rotate(360deg); } }
        .vr-transcript-wrap {
          position: relative;
        }
        .vr-transcript-input {
          width: 100%;
          box-sizing: border-box;
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 14px;
          padding: 12px 14px;
          min-height: 84px;
          max-height: 160px;
          resize: none;
          font-family: inherit;
          font-size: 14px;
          line-height: 1.6;
          color: rgba(255,255,255,0.9);
          outline: none;
          transition: border-color 0.15s, background 0.15s;
        }
        .vr-transcript-input:focus {
          border-color: rgba(99,102,241,0.5);
          background: rgba(255,255,255,0.06);
        }
        .vr-transcript-input::placeholder {
          color: rgba(255,255,255,0.28);
          font-style: italic;
        }
        .vr-actions {
          display: flex; gap: 10px;
        }
        .vr-btn {
          flex: 1; padding: 11px 0; border-radius: 12px;
          font-size: 14px; font-weight: 600; cursor: pointer;
          border: none; transition: opacity 0.15s, transform 0.12s, background 0.15s;
          display: flex; align-items: center; justify-content: center; gap: 6px;
        }
        .vr-btn:active { transform: scale(0.98); }
        .vr-btn-stop {
          background: rgba(239,68,68,0.16);
          border: 1px solid rgba(239,68,68,0.3);
          color: #fca5a5;
        }
        .vr-btn-stop:hover { background: rgba(239,68,68,0.24); color: #fff; }
        .vr-btn-send {
          background: linear-gradient(135deg,#0ea5e9,#6366f1);
          color: #fff;
        }
        .vr-btn-send:disabled { opacity:0.35; cursor:not-allowed; }
        .vr-btn-retry {
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.12);
          color: rgba(255,255,255,0.7);
        }
        .vr-btn-retry:hover { background: rgba(255,255,255,0.12); color:#fff; }
        .vr-error {
          text-align: center;
          font-size: 13px;
          color: #fca5a5;
          background: rgba(239,68,68,0.08);
          border: 1px solid rgba(239,68,68,0.2);
          border-radius: 10px;
          padding: 10px 12px;
        }
      `}</style>

      <div className="vr-overlay" onClick={(e) => { if (e.target === e.currentTarget) handleClose(); }}>
        <div className="vr-modal" role="dialog" aria-label="Voice recorder">
          {/* Header */}
          <div className="vr-header">
            <span className="vr-title">🎙 Voice Message</span>
            <button className="vr-close" onClick={handleClose} aria-label="Close">✕</button>
          </div>

          {/* Waveform bars */}
          <div className="vr-waveform">
            {bars.map((h, i) => (
              <div
                key={i}
                className={`vr-bar ${phase !== "recording" ? "idle" : ""}`}
                style={{ height: `${h}px` }}
              />
            ))}
          </div>

          {/* Status / timer */}
          <div className="vr-status">
            {phase === "recording" && (
              <><div className="vr-dot" /><span>Recording — {fmt(elapsed)}</span></>
            )}
            {phase === "transcribing" && (
              <><div className="vr-spinner" /><span>Transcribing audio…</span></>
            )}
            {phase === "stopped" && <span>Recording complete — {fmt(elapsed)}</span>}
            {phase === "idle"     && <span>Preparing microphone…</span>}
            {phase === "error"    && <span>Microphone error</span>}
          </div>

          {/* Error */}
          {errMsg && <div className="vr-error">{errMsg}</div>}

          {/* Editable live transcript */}
          {phase !== "error" && (
            <div className="vr-transcript-wrap">
              <textarea
                className="vr-transcript-input"
                value={transcript}
                onChange={(e) => {
                  setTranscript(e.target.value);
                  finalTextRef.current = e.target.value;
                }}
                placeholder={
                  phase === "recording"
                    ? "Listening in English… speak into your microphone"
                    : phase === "transcribing"
                    ? "Transcribing with AI…"
                    : "No transcript captured. Type here or click Retry."
                }
                rows={3}
                disabled={phase === "transcribing"}
              />
            </div>
          )}

          {/* Action buttons */}
          <div className="vr-actions">
            {phase === "recording" && (
              <button className="vr-btn vr-btn-stop" onClick={() => { void stopRecording(); }}>
                ⏹ Stop Recording
              </button>
            )}
            {phase === "transcribing" && (
              <button className="vr-btn vr-btn-retry" disabled style={{ opacity: 0.6 }}>
                Processing audio…
              </button>
            )}
            {phase === "stopped" && (
              <>
                <button className="vr-btn vr-btn-retry" onClick={handleRetry}>
                  ↺ Retry
                </button>
                <button className="vr-btn vr-btn-send" onClick={handleSend} disabled={!canSend}>
                  ↑ Send
                </button>
              </>
            )}
            {phase === "error" && (
              <button className="vr-btn vr-btn-retry" onClick={handleRetry}>
                ↺ Try Again
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
