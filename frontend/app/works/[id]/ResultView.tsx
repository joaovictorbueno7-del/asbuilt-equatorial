"use client";

import { useState, useMemo } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";

const Map = dynamic(() => import("./Map"), { ssr: false, loading: () => <MapSkeleton /> });

type Analysis = {
  structure_type: string;
  condition: string;
  non_conformities: string[];
  confidence: number;
  details: string;
};

type Structure = {
  image: string;
  analysis: Analysis;
  placemark?: { name: string; lon: number; lat: number; alt: number };
};

type Work = {
  run_id: string;
  agent_code: string;
  status: string;
  confidence_score: number;
  output: {
    structures?: Structure[];
    non_conformities?: { image: string; issue: string; placemark: string | null }[];
    quality_score?: number;
    image_count?: number;
    placemark_count?: number;
    summary?: string;
  };
  error: string;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
};

type Case = {
  id: string;
  feedback_score: number;
  is_correct: boolean;
  input: { image_key?: string; placemark?: string | null };
  output: Analysis;
};

const STATUS_LABEL: Record<string, string> = {
  pending: "Aguardando", running: "Processando", completed: "Concluída",
  needs_human: "Revisão humana", failed: "Falhou",
};

const COND_BADGE: Record<string, string> = {
  boa: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  regular: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  ruim: "bg-red-500/20 text-red-300 border-red-500/40",
};

export default function ResultView({ work, cases }: { work: Work; cases: Case[] }) {
  const structures = work.output.structures || [];
  const nonConf = work.output.non_conformities || [];
  const quality = work.output.quality_score || 0;

  const caseByImage = useMemo(() => {
    const m: Record<string, Case> = {};
    const list = Array.isArray(cases) ? cases : [];
    list.forEach((c) => { if (c?.input?.image_key) m[c.input.image_key] = c; });
    return m;
  }, [cases]);

  const points = structures
    .filter((s) => s.placemark)
    .map((s) => ({
      lon: s.placemark!.lon, lat: s.placemark!.lat, name: s.placemark!.name,
      type: s.analysis.structure_type, condition: s.analysis.condition,
    }));

  const isProcessing = work.status === "pending" || work.status === "running";

  return (
    <div className="space-y-6">
      <div>
        <Link href="/dashboard" className="text-sm text-slate-400 hover:text-accent">← Dashboard</Link>
        <div className="mt-3 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-2xl font-bold">Resultado da Análise</h1>
            <div className="mt-1 text-sm text-slate-400 font-mono">{work.run_id}</div>
          </div>
          <span className={`text-xs uppercase font-semibold px-3 py-1.5 rounded-md border ${
            work.status === "completed" ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
            : work.status === "needs_human" ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
            : work.status === "failed" ? "bg-red-500/20 text-red-300 border-red-500/40"
            : "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 animate-pulse"
          }`}>
            {STATUS_LABEL[work.status] || work.status}
          </span>
        </div>
      </div>

      {isProcessing && (
        <div className="bg-cyan-950/30 border border-cyan-900/50 rounded-xl p-4 text-cyan-200 text-sm flex items-center gap-3">
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
          Análise em andamento. <button onClick={() => location.reload()} className="underline">Atualizar página</button>
        </div>
      )}

      <div className="grid lg:grid-cols-4 gap-4">
        <ScoreCard quality={quality} />
        <Stat label="Fotos analisadas" value={work.output.image_count || 0} />
        <Stat label="Placemarks" value={work.output.placemark_count || 0} />
        <Stat label="Não-conformidades" value={nonConf.length} alert={nonConf.length > 0} />
      </div>

      {points.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-3">Mapa</h2>
          <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden h-[420px]">
            <Map points={points} />
          </div>
        </section>
      )}

      {nonConf.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-3 text-red-300">Não-conformidades encontradas</h2>
          <div className="bg-red-950/20 border border-red-900/40 rounded-xl divide-y divide-red-900/30">
            {nonConf.map((nc, i) => (
              <div key={i} className="px-4 py-3 flex items-start gap-3">
                <span className="text-red-400 mt-0.5">⚠</span>
                <div className="flex-1">
                  <div className="text-red-100 text-sm">{nc.issue}</div>
                  <div className="text-xs text-red-400/70 mt-0.5 font-mono">
                    {nc.placemark ? `${nc.placemark} · ` : ""}{nc.image}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section>
        <h2 className="text-lg font-semibold mb-3">Estruturas identificadas</h2>
        {structures.length === 0 ? (
          <p className="text-slate-500 text-sm">Nenhuma estrutura analisada ainda.</p>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {structures.map((s, i) => (
              <StructureCard key={i} runId={work.run_id} structure={s} caseInfo={caseByImage[s.image]} />
            ))}
          </div>
        )}
      </section>

      <div className="flex items-center justify-between gap-3 pt-4 border-t border-grid-line flex-wrap">
        <DownloadReportButtons runId={work.run_id} disabled={isProcessing} />
        <div className="flex items-center gap-3">
          <Link
            href="/dashboard"
            className="px-4 py-2 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated transition text-sm"
          >
            Voltar
          </Link>
          <button
            disabled
            title="Agente 02 será implementado em seguida"
            className="px-4 py-2 rounded-lg bg-accent/30 text-bg/70 font-semibold cursor-not-allowed"
          >
            Avançar para Agente 02 →
          </button>
        </div>
      </div>
    </div>
  );
}

function ScoreCard({ quality }: { quality: number }) {
  const color = quality >= 80 ? "text-emerald-400" : quality >= 50 ? "text-amber-400" : "text-red-400";
  const bg = quality >= 80 ? "from-emerald-500/20" : quality >= 50 ? "from-amber-500/20" : "from-red-500/20";
  return (
    <div className={`bg-gradient-to-br ${bg} to-bg-card border border-grid-line rounded-xl p-5 col-span-1`}>
      <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Qualidade</div>
      <div className={`mt-2 text-4xl font-bold ${color}`}>{quality}<span className="text-lg text-slate-500">/100</span></div>
    </div>
  );
}

function Stat({ label, value, alert }: { label: string; value: number; alert?: boolean }) {
  return (
    <div className="bg-bg-card border border-grid-line rounded-xl p-5">
      <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">{label}</div>
      <div className={`mt-2 text-3xl font-bold ${alert ? "text-red-400" : "text-white"}`}>{value}</div>
    </div>
  );
}

function StructureCard({ runId, structure, caseInfo }: {
  runId: string; structure: Structure; caseInfo?: Case;
}) {
  const [feedback, setFeedback] = useState<{ score: number; sent: boolean; correct?: boolean } | null>(
    caseInfo ? { score: caseInfo.feedback_score, sent: false, correct: caseInfo.is_correct } : null,
  );
  const [loading, setLoading] = useState(false);
  const a = structure.analysis;

  async function send(correct: boolean) {
    if (!caseInfo) return;
    setLoading(true);
    try {
      const r = await fetch(`/api/agents/kmz_analyzer/cases/${caseInfo.id}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_correct: correct, notes: "" }),
      });
      if (r.ok) {
        const data = await r.json();
        setFeedback({ score: data.feedback_score, sent: true, correct });
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden">
      <div className="aspect-video bg-bg-elevated relative">
        <img
          src={`/api/works/${runId}/image?key=${encodeURIComponent(structure.image)}`}
          alt={structure.image}
          className="w-full h-full object-cover"
        />
        <span className="absolute top-2 left-2 text-[10px] font-mono bg-bg/80 text-slate-300 px-2 py-0.5 rounded">
          {structure.analysis.structure_type}
        </span>
        <span className={`absolute top-2 right-2 text-[10px] uppercase font-semibold px-2 py-0.5 rounded border ${
          COND_BADGE[a.condition] || "bg-slate-500/20 text-slate-300 border-slate-500/40"
        }`}>
          {a.condition}
        </span>
      </div>
      <div className="p-4 space-y-3">
        {structure.placemark && (
          <div>
            <div className="text-sm font-semibold text-white">{structure.placemark.name}</div>
            <div className="text-xs text-slate-500 font-mono mt-0.5">
              {structure.placemark.lat.toFixed(5)}, {structure.placemark.lon.toFixed(5)}
            </div>
          </div>
        )}
        <p className="text-sm text-slate-300 line-clamp-3">{a.details}</p>
        {a.non_conformities.length > 0 && (
          <ul className="space-y-1">
            {a.non_conformities.map((nc, i) => (
              <li key={i} className="text-xs text-red-300 flex gap-1.5">
                <span>⚠</span><span>{nc}</span>
              </li>
            ))}
          </ul>
        )}
        <div className="flex items-center justify-between pt-2 border-t border-grid-line/50">
          <div className="text-xs text-slate-500">
            confiança IA: <span className="font-mono text-slate-300">{Math.round(a.confidence * 100)}%</span>
          </div>
          {caseInfo && (
            <div className="flex items-center gap-1.5">
              {feedback?.sent && (
                <span className="text-xs text-slate-400">score {feedback.score.toFixed(2)}</span>
              )}
              <button
                disabled={loading}
                onClick={() => send(true)}
                className={`text-xs px-2 py-1 rounded border transition ${
                  feedback?.correct === true
                    ? "bg-emerald-500/30 border-emerald-500/60 text-emerald-200"
                    : "border-grid-line hover:border-emerald-500/50 hover:bg-emerald-500/10 text-slate-400"
                } disabled:opacity-50`}
                title="Marcar como correto"
              >
                ✓
              </button>
              <button
                disabled={loading}
                onClick={() => send(false)}
                className={`text-xs px-2 py-1 rounded border transition ${
                  feedback?.correct === false
                    ? "bg-red-500/30 border-red-500/60 text-red-200"
                    : "border-grid-line hover:border-red-500/50 hover:bg-red-500/10 text-slate-400"
                } disabled:opacity-50`}
                title="Marcar como incorreto"
              >
                ✗
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MapSkeleton() {
  return <div className="w-full h-full flex items-center justify-center text-slate-500 text-sm">Carregando mapa…</div>;
}

function DownloadReportButtons({ runId, disabled }: { runId: string; disabled: boolean }) {
  const [downloading, setDownloading] = useState<"docx" | "pdf" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleDownload(fmt: "docx" | "pdf") {
    setDownloading(fmt);
    setError(null);
    try {
      const res = await fetch(`/api/works/${runId}/report/download?fmt=${fmt}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Erro ${res.status}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `relatorio_${runId.slice(0, 8)}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Erro ao gerar relatório");
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-slate-500 mr-1">Relatório:</span>
      <button
        onClick={() => handleDownload("docx")}
        disabled={disabled || downloading !== null}
        className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-blue-600/20 border border-blue-500/40 text-blue-300 hover:bg-blue-600/30 hover:border-blue-400/60 transition text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {downloading === "docx" ? (
          <span className="w-4 h-4 border-2 border-blue-400/40 border-t-blue-400 rounded-full animate-spin" />
        ) : (
          <span>📄</span>
        )}
        Word (.docx)
      </button>
      <button
        onClick={() => handleDownload("pdf")}
        disabled={disabled || downloading !== null}
        className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-red-600/20 border border-red-500/40 text-red-300 hover:bg-red-600/30 hover:border-red-400/60 transition text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {downloading === "pdf" ? (
          <span className="w-4 h-4 border-2 border-red-400/40 border-t-red-400 rounded-full animate-spin" />
        ) : (
          <span>📕</span>
        )}
        PDF
      </button>
      {error && (
        <span className="text-xs text-red-400 ml-1">⚠ {error}</span>
      )}
    </div>
  );
}
