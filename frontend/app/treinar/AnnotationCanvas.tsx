"use client";

import { useRef, useEffect, useState, useCallback } from "react";

export type BBox = { x: number; y: number; w: number; h: number }; // 0-1 frações

type DetectResult = {
  found: boolean;
  bbox: BBox | null;
  confidence: number;
  description: string;
};

type Props = {
  imageFile: File;
  label: string;           // o que detectar, ex: "MT:N1" ou "POSTE:12/600"
  onAnnotated: (bbox: BBox | null) => void;
};

type DrawMode = "idle" | "drawing" | "dragging" | "confirmed";

export default function AnnotationCanvas({ imageFile, label, onAnnotated }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [detecting, setDetecting] = useState(false);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [bbox, setBbox] = useState<BBox | null>(null);          // em fração 0-1
  const [mode, setMode] = useState<DrawMode>("idle");
  const [error, setError] = useState<string | null>(null);

  // Estado de drag (para redesenhar)
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const currentBbox = useRef<BBox | null>(null);

  // Carrega imagem no canvas
  const imgUrl = useRef<string>("");

  useEffect(() => {
    if (imgUrl.current) URL.revokeObjectURL(imgUrl.current);
    imgUrl.current = URL.createObjectURL(imageFile);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      drawCanvas(null);
    };
    img.src = imgUrl.current;
    return () => { URL.revokeObjectURL(imgUrl.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageFile]);

  const drawCanvas = useCallback((box: BBox | null, color = "#22d3ee", dashed = true) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const container = containerRef.current;
    const cw = container?.clientWidth || 600;
    const scale = cw / img.naturalWidth;
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;

    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    if (box) {
      const px = box.x * canvas.width;
      const py = box.y * canvas.height;
      const pw = box.w * canvas.width;
      const ph = box.h * canvas.height;

      // Overlay escuro fora da box
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.clearRect(px, py, pw, ph);
      ctx.drawImage(img, px / scale, py / scale, pw / scale, ph / scale, px, py, pw, ph);

      // Borda
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      if (dashed) ctx.setLineDash([8, 4]);
      else ctx.setLineDash([]);
      ctx.strokeRect(px, py, pw, ph);

      // Rótulo
      const label_text = box === currentBbox.current ? "arrastando…" : "";
      if (label_text) {
        ctx.fillStyle = color;
        ctx.font = "bold 13px monospace";
        ctx.fillText(label_text, px + 4, py - 6);
      }

      // Cantos
      const corner = 8;
      ctx.fillStyle = color;
      ctx.setLineDash([]);
      [[px, py], [px + pw, py], [px, py + ph], [px + pw, py + ph]].forEach(([cx, cy]) => {
        ctx.beginPath();
        ctx.arc(cx, cy, corner / 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, []);

  // Redesenha quando bbox muda
  useEffect(() => {
    if (!detecting) {
      const color = mode === "confirmed" ? "#10b981" : "#22d3ee";
      const dashed = mode !== "confirmed";
      drawCanvas(bbox, color, dashed);
    }
  }, [bbox, mode, detecting, drawCanvas]);

  // ── Detecção automática ───────────────────────────────────────────────────
  async function detect() {
    if (!imageFile || !label.trim()) return;
    setDetecting(true);
    setError(null);
    setResult(null);
    setBbox(null);
    setMode("idle");
    onAnnotated(null);

    const fd = new FormData();
    fd.append("file", imageFile);
    fd.append("label", label);

    try {
      const r = await fetch("/api/learning/detect", { method: "POST", body: fd });
      const data: DetectResult = await r.json();
      setResult(data);
      if (data.found && data.bbox) {
        setBbox(data.bbox);
        currentBbox.current = data.bbox;
        setMode("idle");
      }
    } catch {
      setError("Falha na detecção");
    } finally {
      setDetecting(false);
    }
  }

  // ── Drag para redesenhar ──────────────────────────────────────────────────
  function canvasPos(e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
    };
  }

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (mode === "confirmed") return;
    dragStart.current = canvasPos(e);
    setMode("drawing");
  }

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (mode !== "drawing" || !dragStart.current) return;
    const pos = canvasPos(e);
    const newBox: BBox = {
      x: Math.min(dragStart.current.x, pos.x),
      y: Math.min(dragStart.current.y, pos.y),
      w: Math.abs(pos.x - dragStart.current.x),
      h: Math.abs(pos.y - dragStart.current.y),
    };
    currentBbox.current = newBox;
    setBbox(newBox);
  }

  function onMouseUp() {
    if (mode !== "drawing") return;
    setMode("idle");
    dragStart.current = null;
  }

  // ── Confirmação ───────────────────────────────────────────────────────────
  function confirm() {
    setMode("confirmed");
    onAnnotated(bbox);
  }

  function reject() {
    setBbox(null);
    currentBbox.current = null;
    setResult(null);
    setMode("idle");
    onAnnotated(null);
  }

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const confColor = confidencePct >= 80 ? "text-emerald-400" : confidencePct >= 50 ? "text-amber-400" : "text-red-400";

  return (
    <div className="space-y-3">
      {/* Botão detectar */}
      <button
        type="button"
        onClick={detect}
        disabled={detecting || !label.trim()}
        className="w-full flex items-center justify-center gap-2 py-2 rounded-xl border-2 border-accent/40 bg-accent/10 text-accent font-semibold text-sm hover:bg-accent/20 transition disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {detecting ? (
          <>
            <span className="w-4 h-4 border-2 border-accent/40 border-t-accent rounded-full animate-spin" />
            Detectando com IA…
          </>
        ) : (
          <>🔍 Detectar na foto</>
        )}
      </button>

      {/* Canvas interativo */}
      <div ref={containerRef} className="relative rounded-xl overflow-hidden bg-black">
        <canvas
          ref={canvasRef}
          className={`w-full block ${mode === "confirmed" ? "cursor-default" : "cursor-crosshair"}`}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        />
        {!bbox && !detecting && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-xs text-slate-500 bg-black/60 px-3 py-1 rounded-full">
              {result && !result.found ? "Não encontrado — arraste para marcar manualmente" : "Clique em Detectar ou arraste para marcar"}
            </span>
          </div>
        )}
      </div>

      {/* Resultado da detecção */}
      {result && bbox && mode !== "confirmed" && (
        <div className="rounded-xl border border-grid-line bg-bg-elevated p-3 space-y-2">
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-0.5">
              <div className="text-xs font-semibold text-white">
                {result.found ? "✓ Elemento localizado" : "✗ Não encontrado automaticamente"}
              </div>
              <div className="text-xs text-slate-400 italic">{result.description}</div>
            </div>
            {result.found && (
              <span className={`text-xs font-mono font-bold flex-shrink-0 ${confColor}`}>
                {confidencePct}%
              </span>
            )}
          </div>

          <div className="text-[11px] text-slate-500">
            {mode === "drawing"
              ? "Arraste para corrigir a área…"
              : "Arraste na foto para reposicionar a caixa"}
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={confirm}
              className="py-2 rounded-lg bg-emerald-600/20 border border-emerald-500/40 text-emerald-300 font-semibold text-sm hover:bg-emerald-600/30 transition"
            >
              ✓ Correto
            </button>
            <button
              type="button"
              onClick={reject}
              className="py-2 rounded-lg bg-red-600/20 border border-red-500/40 text-red-300 font-semibold text-sm hover:bg-red-600/30 transition"
            >
              ✗ Errou — limpar
            </button>
          </div>
        </div>
      )}

      {/* Confirmado */}
      {mode === "confirmed" && bbox && (
        <div className="rounded-xl border border-emerald-500/40 bg-emerald-950/30 p-3 flex items-center justify-between">
          <div>
            <div className="text-xs font-semibold text-emerald-300">✓ Anotação confirmada</div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              x:{(bbox.x * 100).toFixed(0)}% y:{(bbox.y * 100).toFixed(0)}%
              &nbsp;·&nbsp;
              {(bbox.w * 100).toFixed(0)}×{(bbox.h * 100).toFixed(0)}%
            </div>
          </div>
          <button
            type="button"
            onClick={reject}
            className="text-xs text-slate-500 hover:text-red-400 transition"
          >
            Refazer
          </button>
        </div>
      )}

      {error && (
        <div className="text-xs text-red-400 text-center">{error}</div>
      )}
    </div>
  );
}
