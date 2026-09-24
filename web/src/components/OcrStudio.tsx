"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type OcrResponse = {
  text: string;
  filename: string | null;
  characters: number;
  model: string;
  model_id: string;
  model_label: string;
  device: string;
};

type ModelInfo = {
  id: string;
  label: string;
  description: string;
  hf_id: string;
  languages: string[];
  note: string;
  ready: boolean;
  active: boolean;
};

type Stage = "idle" | "ready" | "working" | "done" | "error";

const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "";

const FALLBACK_MODELS: ModelInfo[] = [
  {
    id: "trocr-handwritten",
    label: "TrOCR Handwritten",
    description: "Microsoft TrOCR base — handwritten English (IAM)",
    hf_id: "microsoft/trocr-base-handwritten",
    languages: ["en"],
    note: "Best on single-line handwriting crops.",
    ready: false,
    active: false,
  },
];

export default function OcrStudio() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [dragOver, setDragOver] = useState(false);
  const [modelId, setModelId] = useState("trocr-handwritten");
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<{
    characters: number;
    device: string;
    model_label: string;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/models`);
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        if (Array.isArray(data.models) && data.models.length) {
          setModels(data.models);
          if (data.default) setModelId(data.default);
        }
      } catch {
        /* keep fallback list */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedModel = useMemo(
    () => models.find((m) => m.id === modelId) ?? models[0],
    [models, modelId],
  );

  const acceptFile = useCallback((next: File | null) => {
    if (!next) return;
    if (!next.type.startsWith("image/") && !/\.(png|jpe?g|webp|tiff?|bmp)$/i.test(next.name)) {
      setError("Please drop an image file (PNG, JPEG, WEBP, TIFF).");
      setStage("error");
      return;
    }
    setError(null);
    setText("");
    setMeta(null);
    setFile(next);
    setStage("ready");
    const url = URL.createObjectURL(next);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragOver(false);
      const dropped = event.dataTransfer.files?.[0];
      if (dropped) acceptFile(dropped);
    },
    [acceptFile],
  );

  const runOcr = useCallback(async () => {
    if (!file) return;
    setStage("working");
    setError(null);
    setText("");

    const body = new FormData();
    body.append("file", file);
    body.append("language", "en");
    body.append("model", modelId);

    try {
      const controller = new AbortController();
      const timeoutMs = 30 * 60 * 1000;
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      const res = await fetch(`${API_BASE}/api/ocr`, {
        method: "POST",
        body,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const payload = await res.json();
          detail =
            typeof payload.detail === "string"
              ? payload.detail
              : JSON.stringify(payload.detail) || detail;
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }

      const data = (await res.json()) as OcrResponse;
      setText(data.text || "");
      setMeta({
        characters: data.characters,
        device: data.device,
        model_label: data.model_label || data.model,
      });
      setStage("done");
    } catch (err) {
      let message = err instanceof Error ? err.message : "Something went wrong.";
      if (err instanceof Error && err.name === "AbortError") {
        message = "OCR timed out. Try a smaller single-line crop or a GPU.";
      } else if (
        message.includes("Failed to fetch") ||
        message.includes("NetworkError") ||
        message.includes("ECONNREFUSED")
      ) {
        message =
          "Cannot reach the OCR API at http://127.0.0.1:8000. Start it with scripts\\start-api.bat.";
      }
      setError(message);
      setStage("error");
    }
  }, [file, modelId]);

  const downloadTxt = useCallback(() => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const base = file?.name?.replace(/\.[^.]+$/, "") || "katib-ocr";
    a.href = url;
    a.download = `${base}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [file, text]);

  const reset = useCallback(() => {
    setFile(null);
    setText("");
    setError(null);
    setMeta(null);
    setStage("idle");
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  return (
    <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-5xl flex-col px-5 py-10 sm:px-8 sm:py-14">
      <header className="animate-rise mb-10 text-center sm:mb-14">
        <p className="font-[family-name:var(--font-naskh)] mb-3 text-sm tracking-[0.2em] text-[var(--teal-bright)]">
          كاتب
        </p>
        <h1 className="font-[family-name:var(--font-fraunces)] text-[clamp(3.4rem,12vw,6.5rem)] leading-[0.9] tracking-[-0.03em] text-[var(--paper)]">
          Katib
        </h1>
        <p className="animate-rise-delay mx-auto mt-5 max-w-md text-base text-[var(--mist-dim)] sm:text-lg">
          Drop a handwritten line. Extract the text. Download a `.txt`.
        </p>
      </header>

      <main className="animate-rise-delay-2 flex flex-1 flex-col gap-6">
        <div className="mx-auto w-full max-w-xl">
          <label
            htmlFor="model-select"
            className="mb-2 block text-xs tracking-[0.14em] uppercase text-[var(--mist-dim)]"
          >
            Model
          </label>
          <select
            id="model-select"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            disabled={stage === "working"}
            className="w-full appearance-none rounded-[2px] border border-[var(--line)] bg-[var(--ink-soft)] px-4 py-3 text-[var(--paper)] outline-none transition focus:border-[var(--teal-bright)]"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          {selectedModel && (
            <p className="mt-2 text-sm text-[var(--mist-dim)]">
              {selectedModel.description}
              {selectedModel.note ? ` — ${selectedModel.note}` : ""}
            </p>
          )}
        </div>

        {!preview ? (
          <div
            className="dropzone relative flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-[2px] border border-dashed border-[var(--line)] bg-[color-mix(in_oklab,var(--ink-soft)_70%,transparent)] px-6 py-16 text-center sm:min-h-[360px]"
            data-active={dragOver}
            onDragEnter={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setDragOver(false);
            }}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
            }}
          >
            <span className="animate-pulse-ring pointer-events-none absolute h-28 w-28 rounded-full border border-[var(--teal-bright)]/40" />
            <span className="mb-5 font-[family-name:var(--font-fraunces)] text-5xl text-[var(--teal-bright)]">
              +
            </span>
            <p className="text-lg text-[var(--paper)]">Place your file here</p>
            <p className="mt-2 text-sm text-[var(--mist-dim)]">
              PNG, JPEG, WEBP, TIFF — up to 25 MB
            </p>
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp,.tif,.tiff"
              className="hidden"
              onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
            />
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
            <div className="flex flex-col gap-4">
              <div className="relative overflow-hidden rounded-[2px] border border-[var(--line)] bg-[var(--ink-soft)]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={preview}
                  alt="Upload preview"
                  className="max-h-[420px] w-full object-contain"
                />
                {stage === "working" && (
                  <div className="animate-shimmer absolute inset-0 bg-[color-mix(in_oklab,var(--ink)_35%,transparent)]" />
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={runOcr}
                  disabled={stage === "working"}
                  className="rounded-[2px] bg-[var(--paper)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)] transition hover:bg-white disabled:cursor-wait disabled:opacity-60"
                >
                  {stage === "working" ? "Reading…" : "Extract text"}
                </button>

                <button
                  type="button"
                  onClick={reset}
                  className="px-3 py-2 text-sm text-[var(--mist-dim)] transition hover:text-[var(--paper)]"
                >
                  New file
                </button>
              </div>
            </div>

            <div className="flex min-h-[280px] flex-col border-t border-[var(--line)] pt-5 lg:border-t-0 lg:border-l lg:pl-6 lg:pt-0">
              <div className="mb-3 flex items-center justify-between gap-3">
                <h2 className="font-[family-name:var(--font-fraunces)] text-2xl text-[var(--paper)]">
                  Result
                </h2>
                {stage === "done" && text && (
                  <button
                    type="button"
                    onClick={downloadTxt}
                    className="rounded-[2px] border border-[var(--teal-bright)]/50 px-4 py-2 text-sm text-[var(--teal-bright)] transition hover:bg-[color-mix(in_oklab,var(--teal)_18%,transparent)]"
                  >
                    Download .txt
                  </button>
                )}
              </div>

              {stage === "working" && (
                <p className="text-sm text-[var(--mist-dim)]">
                  Loading / running <span className="text-[var(--paper)]">{selectedModel?.label}</span>
                  … First use downloads weights from Hugging Face.
                </p>
              )}

              {error && (
                <p className="text-sm text-red-300" role="alert">
                  {error}
                </p>
              )}

              {stage === "done" && (
                <>
                  <div
                    dir="ltr"
                    className="result-scroll max-h-[420px] flex-1 overflow-y-auto whitespace-pre-wrap font-[family-name:var(--font-figtree)] text-[1.05rem] leading-8 text-[var(--paper)]"
                  >
                    {text || "No text detected."}
                  </div>
                  {meta && (
                    <p className="mt-4 text-xs tracking-wide text-[var(--mist-dim)]">
                      {meta.characters.toLocaleString()} characters · {meta.model_label} ·{" "}
                      {meta.device}
                    </p>
                  )}
                </>
              )}

              {(stage === "ready" || stage === "idle") && !error && (
                <p className="text-sm text-[var(--mist-dim)]">
                  Works best on a single handwritten English line crop.
                </p>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="mt-12 text-center text-xs text-[var(--mist-dim)]/80">
        Local TrOCR · microsoft/trocr-base-handwritten
      </footer>
    </div>
  );
}
