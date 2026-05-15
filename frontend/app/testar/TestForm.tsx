"use client";

import { useState, useRef, useEffect, DragEvent } from "react";

type RecognizeResult = {
  tipo: string;
  codigo: string | null;
  tamanho: string | null;
  conformidade: boolean | null;
  confianca: number;
  descricao: string;
  observacoes: string;
  n_cases_used: number;
};

const TIPO_LABEL: Record<string, { icon: string; label: string; color: string }> = {
  poste:          { icon: "🪵", label: "Poste",          color: "text-amber-300" },
  estrutura_mt:   { icon: "⚡", label: "Estrutura MT",   color: "text-orange-300" },
  estrutura_bt:   { icon: "⚡", label: "Estrutura BT",   color: "text-blue-300" },
  estrutura_mt_bt:{ icon: "⚡", label: "Estrutura MT+BT",color: "text-purple-300" },
  desconhecido:   { icon: "❓", label: "Não identificado",color: "text-slate-400" },
};

export default function TestForm() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile]       = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [pasteHint, setPasteHint] = useState(false);

  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult]       = useState<RecognizeResult | null>(null);
  const [error, setError]         = useState<string | null>(null);

  // ── Ctrl+V global ───────────────────────────────────────────────────────────
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of Array.from(items)) {
        if (item.type.startsWith("image/")) {
          const blob = item.getAsFile();
          if (blob) { pickBlob(blob); setPasteHint(true); setTimeout(() => setPasteHint(false), 1500); }
          break;
        }
      }
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function pickBlob(blob: Blob) {
    if (blob.size > 10 * 1024 * 1024) { setError("Imagem supera 10 MB"); return; }
    setError(null); setResult(null);
    const f = new File([blob], "foto_teste.jpg", { type: blob.type || "image/jpeg" });
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(blob));
  }

  function pickFile(f: File | null) {
    if (!f) return;
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["jpg","jpeg","png"].includes(ext || "")) { setError("Use JPG ou PNG"); return; }
    if (f.size > 10 * 1024 * 1024) { setError("Arquivo supera 10 MB"); return; }
    setError(null); setResult(null);
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
  }

  function onDrop(e: DragEvent) {
    e.preventDefault(); setDragOver(false);
    pickFile(e.dataTransfer.files?.[0] || null);
  }

  function clear() {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  // ── Análise automática quando file muda ─────────────────────────────────────
  useEffect(() => {
    if (file) analyze(file);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  async function analyze(f: File) {
    setAnalyzing(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch("/api/learning/recognize", { method: "POST", body: fd });
      const data: RecognizeResult = await r.json();
      if (!r.ok) { setError((data as { error?: string }).error || "Erro na análise"); return; }
      setResult(data);
    } catch {
      setError("Falha de rede");
    } finally {
      setAnalyzing(false);
    }
  }

  const tipoInfo = result ? (TIPO_LABEL[result.tipo] ?? TIPO_LABEL.desconhecido) : null;
  const pct = result ? Math.round(result.confianca * 100) : 0;
  const confColor = pct >= 80 ? "text-emerald-400" : pct >= 55 ? "text-amber-400" : "text-red-400";
  const confBarColor = pct >= 80 ? "bg-emerald-500" : pct >= 55 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="space-y-6">
      {/* ── Drop zone / Preview ─────────────────────────────────────────────── */}
      <div className="bg-bg-card border border-grid-line rounded-2xl p-6">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
            Foto para testar
          </span>
          <span className={`text-xs font-mono px-2 py-0.5 rounded border transition-all duration-300 ${
            pasteHint
              ? "bg-accent/20 text-accent border-accent/40 scale-105"
              : "bg-bg-elevated text-slate-500 border-grid-line"
          }`}>
            Ctrl+V para colar
          </span>
        </div>

        <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" className="hidden"
          onChange={(e) => pickFile(e.target.files?.[0] || null)} />

        {preview ? (
          <div className="relative rounded-xl overflow-hidden bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="foto" className="w-full max-h-80 object-contain" />
            <button
              onClick={clear}
              className="absolute top-2 right-2 bg-black/60 hover:bg-red-900/80 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm transition"
            >
              ×
            </button>
            {pasteHint && (
              <div className="absolute inset-0 flex items-center justify-center bg-accent/10">
                <span className="text-accent font-bold text-sm bg-black/60 px-4 py-2 rounded-xl">✓ Imagem colada!</span>
              </div>
            )}
          </div>
        ) : (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`cursor-pointer rounded-xl border-2 border-dashed transition flex flex-col items-center justify-center gap-4 py-16 ${
              pasteHint
                ? "border-accent bg-accent/5"
                : dragOver
                ? "border-accent bg-accent/5"
                : "border-grid-line bg-bg-elevated hover:border-slate-500"
            }`}
          >
            <div className="text-6xl opacity-20">📸</div>
            <div className="text-center">
              <div className="font-semibold text-slate-200 text-sm mb-3">
                Cole, arraste ou clique para selecionar uma foto
              </div>
              <div className="inline-flex items-center gap-2 bg-bg-card border border-grid-line rounded-lg px-4 py-2">
                <kbd className="text-xs font-mono text-accent font-bold">Ctrl</kbd>
                <span className="text-slate-600 text-xs">+</span>
                <kbd className="text-xs font-mono text-accent font-bold">V</kbd>
                <span className="text-xs text-slate-400 ml-1">para colar do clipboard</span>
              </div>
              <div className="text-xs text-slate-600 mt-2">A análise começa automaticamente</div>
            </div>
          </div>
        )}
      </div>

      {/* ── Analisando ─────────────────────────────────────────────────────── */}
      {analyzing && (
        <div className="bg-bg-card border border-grid-line rounded-2xl p-8 flex flex-col items-center gap-4">
          <div className="relative w-16 h-16">
            <div className="absolute inset-0 rounded-full border-4 border-accent/20" />
            <div className="absolute inset-0 rounded-full border-4 border-t-accent animate-spin" />
            <div className="absolute inset-0 flex items-center justify-center text-2xl">🔍</div>
          </div>
          <div className="text-center">
            <div className="text-white font-semibold">Analisando com IA…</div>
            <div className="text-slate-400 text-xs mt-1">
              Comparando com os casos de treinamento
            </div>
          </div>
        </div>
      )}

      {/* ── Resultado ──────────────────────────────────────────────────────── */}
      {result && !analyzing && (
        <div className="bg-bg-card border border-grid-line rounded-2xl overflow-hidden">
          {/* Header do resultado */}
          <div className="px-6 pt-5 pb-4 border-b border-grid-line flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-3xl">{tipoInfo?.icon}</span>
              <div>
                <div className={`text-lg font-bold ${tipoInfo?.color}`}>
                  {tipoInfo?.label}
                </div>
                {result.codigo && (
                  <div className="text-white font-mono text-sm mt-0.5">
                    {result.tipo === "poste" ? "Nº " : ""}{result.codigo}
                    {result.tamanho && <span className="text-slate-400 ml-2">· {result.tamanho}</span>}
                  </div>
                )}
              </div>
            </div>

            {/* Conformidade */}
            <div className="flex-shrink-0 text-right">
              {result.conformidade === true && (
                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-950/50 border border-emerald-700/40 text-emerald-300 font-semibold text-sm">
                  ✅ Conforme
                </div>
              )}
              {result.conformidade === false && (
                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-950/50 border border-red-700/40 text-red-300 font-semibold text-sm">
                  ❌ Não conforme
                </div>
              )}
              {result.conformidade === null && (
                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/50 border border-slate-700/40 text-slate-400 font-semibold text-sm">
                  — Sem avaliação
                </div>
              )}
            </div>
          </div>

          {/* Body */}
          <div className="px-6 py-5 space-y-5">
            {/* Confiança */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs uppercase tracking-wider text-slate-500 font-medium">Confiança</span>
                <span className={`text-sm font-bold font-mono ${confColor}`}>{pct}%</span>
              </div>
              <div className="h-2 rounded-full bg-bg-elevated overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${confBarColor}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="text-[11px] text-slate-600 mt-1">
                {pct >= 80 ? "Alta confiança — resultado confiável"
                  : pct >= 55 ? "Confiança média — revise a foto se necessário"
                  : "Baixa confiança — foto pode estar ruim ou elemento não reconhecido"}
              </div>
            </div>

            {/* Descrição */}
            {result.descricao && (
              <div>
                <div className="text-xs uppercase tracking-wider text-slate-500 font-medium mb-1">O que foi visto</div>
                <div className="text-sm text-slate-300 bg-bg-elevated rounded-lg px-3 py-2.5">
                  {result.descricao}
                </div>
              </div>
            )}

            {/* Observações */}
            {result.observacoes && (
              <div>
                <div className="text-xs uppercase tracking-wider text-slate-500 font-medium mb-1">Observações</div>
                <div className="text-sm text-slate-400 bg-bg-elevated rounded-lg px-3 py-2.5 italic">
                  {result.observacoes}
                </div>
              </div>
            )}

            {/* Rodapé: casos usados + analisar de novo */}
            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-600">
                  🎓 {result.n_cases_used} caso{result.n_cases_used !== 1 ? "s" : ""} de treinamento usados
                </span>
                {result.n_cases_used === 0 && (
                  <a href="/treinar" className="text-xs text-accent hover:underline">
                    → Adicionar casos
                  </a>
                )}
              </div>
              <button
                onClick={() => file && analyze(file)}
                className="text-xs text-accent hover:underline"
              >
                Analisar novamente
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Erro ───────────────────────────────────────────────────────────── */}
      {error && (
        <div className="rounded-xl bg-red-950/40 border border-red-900/60 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* ── Instrução inicial ───────────────────────────────────────────────── */}
      {!file && !analyzing && !result && (
        <div className="rounded-xl bg-bg-card border border-grid-line border-dashed p-6 text-center space-y-2">
          <div className="text-3xl opacity-30">🤖</div>
          <div className="text-sm text-slate-400">
            Cole uma foto e o agente dirá automaticamente o que reconheceu
          </div>
          <div className="text-xs text-slate-600">
            Quanto mais casos você treinar em{" "}
            <a href="/treinar" className="text-accent hover:underline">Treinar</a>
            , mais preciso fica o reconhecimento
          </div>
        </div>
      )}
    </div>
  );
}
