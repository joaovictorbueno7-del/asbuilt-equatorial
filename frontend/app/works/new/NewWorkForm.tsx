"use client";

import { useState, useRef, DragEvent, FormEvent, useEffect } from "react";
import { useRouter } from "next/navigation";

const CONCESSIONARIAS = ["Equatorial", "Cemig", "Copel", "Enel", "Light", "EDP", "Energisa", "Outro"];
const TIPOS = [
  { value: "as_built", label: "AS BUILT" },
  { value: "obras", label: "Obras" },
  { value: "manutencao", label: "Manutenção" },
];

type Stage = "form" | "uploading" | "processing" | "done" | "error";

type PhotosProgress = { current: number; total: number };

type WorkStatus = {
  status: string;
  output?: { image_count?: number; placemark_count?: number; quality_score?: number };
  // Progresso do kmz_analyzer em tempo real (via agents[].output_summary.photos_progress)
  agents?: Array<{
    agent_code: string;
    status: string;
    output_summary?: { photos_progress?: PhotosProgress; image_count?: number } | null;
  }>;
};

export default function NewWorkForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [workName, setWorkName] = useState("");
  const [municipio, setMunicipio] = useState("");
  const [concessionaria, setConcessionaria] = useState(CONCESSIONARIAS[0]);
  const [tipo, setTipo] = useState(TIPOS[0].value);
  const [stage, setStage] = useState<Stage>("form");
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState<WorkStatus | null>(null);

  function pickFile(f: File | null) {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".kmz")) {
      setError("Arquivo precisa ter extensão .kmz");
      return;
    }
    if (f.size > 100 * 1024 * 1024) {
      setError("Arquivo passa de 100MB");
      return;
    }
    setError(null);
    setFile(f);
    if (!workName) setWorkName(f.name.replace(/\.kmz$/i, ""));
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    pickFile(e.dataTransfer.files?.[0] || null);
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setStage("uploading");
    setError(null);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("work_name", workName);
    fd.append("concessionaria", concessionaria);
    fd.append("tipo", tipo);
    fd.append("municipio", municipio);

    try {
      const r = await fetch("/api/pipelines", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) {
        setError(data.detail || data.error || "Erro no upload");
        setStage("error");
        return;
      }
      setRunId(data.id);
      setStage("processing");
    } catch {
      setError("Falha de rede");
      setStage("error");
    }
  }

  useEffect(() => {
    if (stage !== "processing" || !runId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch(`/api/pipelines/${runId}`);
        const data: WorkStatus = await r.json();
        if (cancelled) return;
        setProgress(data);
        if (["completed", "needs_human", "failed"].includes(data.status)) {
          setStage("done");
          setTimeout(() => router.push(`/pipelines/${runId}`), 1200);
          return;
        }
      } catch {}
      if (!cancelled) setTimeout(tick, 2000);
    };
    tick();
    return () => { cancelled = true; };
  }, [stage, runId, router]);

  if (stage === "processing" || stage === "done" || stage === "uploading") {
    return <ProgressView stage={stage} progress={progress} fileName={file?.name || ""} />;
  }

  return (
    <form onSubmit={submit} className="space-y-6">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition ${
          dragOver ? "border-accent bg-accent/5" : "border-grid-line bg-bg-card hover:border-slate-600"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".kmz"
          className="hidden"
          onChange={(e) => pickFile(e.target.files?.[0] || null)}
        />
        {file ? (
          <div>
            <div className="text-4xl mb-3">📦</div>
            <div className="font-semibold text-white">{file.name}</div>
            <div className="text-xs text-slate-400 mt-1">{(file.size / 1024).toFixed(1)} KB</div>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); setFile(null); }}
              className="mt-3 text-xs text-slate-500 hover:text-red-400"
            >
              Remover
            </button>
          </div>
        ) : (
          <div>
            <div className="text-5xl mb-3 opacity-40">⬆️</div>
            <div className="font-semibold text-slate-200">Arraste um arquivo .KMZ aqui</div>
            <div className="text-sm text-slate-500 mt-1">ou clique para escolher</div>
            <div className="text-xs text-slate-600 mt-3">máx. 100 MB</div>
          </div>
        )}
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <Field label="Nome da obra">
          <input
            value={workName}
            onChange={(e) => setWorkName(e.target.value)}
            required
            className="form-input"
            placeholder="ex: Rede Bairro Cidade Nova"
          />
        </Field>
        <Field label="Município">
          <input
            value={municipio}
            onChange={(e) => setMunicipio(e.target.value)}
            className="form-input"
            placeholder="ex: Fortaleza"
          />
        </Field>
        <Field label="Concessionária">
          <select value={concessionaria} onChange={(e) => setConcessionaria(e.target.value)} className="form-input">
            {CONCESSIONARIAS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Tipo">
          <select value={tipo} onChange={(e) => setTipo(e.target.value)} className="form-input">
            {TIPOS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
        </Field>
      </div>

      {error && (
        <div className="rounded-lg bg-red-950/40 border border-red-900/60 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={!file}
        className="w-full bg-accent hover:bg-accent-hover disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-bg font-bold py-3 rounded-xl transition"
      >
        Iniciar Análise
      </button>

      <style jsx global>{`
        .form-input {
          width: 100%;
          background-color: #0a0e1a;
          border: 1px solid #1e293b;
          color: #e5e7eb;
          padding: 0.625rem 0.875rem;
          border-radius: 0.5rem;
          outline: none;
          transition: border-color 0.15s, box-shadow 0.15s;
        }
        .form-input:focus {
          border-color: #22d3ee;
          box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15);
        }
      `}</style>
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-1.5 block">{label}</span>
      {children}
    </label>
  );
}

function ProgressView({ stage, progress, fileName }: { stage: Stage; progress: WorkStatus | null; fileName: string }) {
  // Pega progresso do kmz_analyzer em tempo real
  const kmzAgent = progress?.agents?.find(a => a.agent_code === "kmz_analyzer");
  const photoProg: PhotosProgress | null =
    kmzAgent?.output_summary?.photos_progress ?? null;
  const photosDone = progress?.status === "completed" || progress?.status === "needs_human";

  const steps = [
    { key: "upload", label: "KMZ recebido", done: true },
    { key: "extract", label: "Extraindo fotos e coordenadas", done: !!progress },
    { key: "analyze", label: "Analisando fotos com Claude Vision", done: photosDone },
    { key: "validate", label: "Validando padrões técnicos", done: stage === "done" },
    { key: "done", label: "Análise concluída", done: stage === "done" },
  ];
  const isFailed = progress?.status === "failed";

  return (
    <div className="bg-bg-card border border-grid-line rounded-2xl p-8 space-y-6">
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-500 mb-1">Processando</div>
        <div className="font-semibold text-white truncate">{fileName}</div>
      </div>
      <div className="space-y-4">
        {steps.map((s, i) => {
          const active = !s.done && (i === 0 || steps[i - 1].done);
          const isVision = s.key === "analyze";
          const pct = photoProg && photoProg.total > 0
            ? Math.round((photoProg.current / photoProg.total) * 100)
            : null;

          return (
            <div key={s.key}>
              <div className="flex items-center gap-3">
                <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-xs ${
                  s.done ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : active ? "bg-accent/20 text-accent border border-accent/40 animate-pulse"
                  : "bg-bg-elevated text-slate-600 border border-grid-line"
                }`}>
                  {s.done ? "✓" : active ? "●" : "○"}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className={s.done ? "text-slate-200" : active ? "text-white font-medium" : "text-slate-600"}>
                      {s.label}
                      {isVision && photoProg && !s.done && (
                        <span className="ml-2 text-xs text-slate-400 font-normal">
                          {photoProg.current}/{photoProg.total} fotos
                        </span>
                      )}
                      {isVision && s.done && kmzAgent?.output_summary?.image_count && (
                        <span className="ml-2 text-xs text-slate-400 font-normal">
                          {kmzAgent.output_summary.image_count} fotos
                        </span>
                      )}
                    </span>
                    {isVision && pct !== null && !s.done && (
                      <span className="text-xs font-mono text-accent flex-shrink-0">{pct}%</span>
                    )}
                  </div>
                  {/* Barra de progresso — só aparece quando Vision está ativa */}
                  {isVision && active && photoProg && photoProg.total > 0 && !s.done && (
                    <div className="mt-1.5 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                      <div
                        className="h-full bg-accent rounded-full transition-all duration-300"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  )}
                  {/* Barra indeterminada quando Vision está ativa mas progresso ainda não chegou */}
                  {isVision && active && !photoProg && (
                    <div className="mt-1.5 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                      <div className="h-full w-1/3 bg-accent/60 rounded-full animate-pulse" />
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {isFailed && (
        <div className="rounded-lg bg-red-950/40 border border-red-900/60 px-4 py-3 text-sm text-red-300">
          Processamento falhou. Veja detalhes no resultado.
        </div>
      )}
      {stage === "done" && (
        <div className="text-center text-sm text-slate-400">Redirecionando para o resultado…</div>
      )}
    </div>
  );
}
