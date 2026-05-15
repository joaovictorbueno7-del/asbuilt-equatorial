"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

function PipelineDownloadButtons({ runId }: { runId: string }) {
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
      setError(e instanceof Error ? e.message : "Erro ao gerar");
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-slate-500">Relatório:</span>
      <button
        onClick={() => handleDownload("docx")}
        disabled={downloading !== null}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600/20 border border-blue-500/40 text-blue-300 hover:bg-blue-600/30 transition text-xs font-medium disabled:opacity-40"
      >
        {downloading === "docx"
          ? <span className="w-3 h-3 border-2 border-blue-400/40 border-t-blue-400 rounded-full animate-spin" />
          : "📄"} Word
      </button>
      <button
        onClick={() => handleDownload("pdf")}
        disabled={downloading !== null}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-600/20 border border-red-500/40 text-red-300 hover:bg-red-600/30 transition text-xs font-medium disabled:opacity-40"
      >
        {downloading === "pdf"
          ? <span className="w-3 h-3 border-2 border-red-400/40 border-t-red-400 rounded-full animate-spin" />
          : "📕"} PDF
      </button>
      {error && <span className="text-xs text-red-400">⚠ {error}</span>}
    </div>
  );
}

type Agent = {
  agent_code: string;
  depends_on: string[];
  run_id: string | null;
  status: string;
  confidence_score: number;
  started_at: string | null;
  finished_at: string | null;
  error: string;
  output_summary: Record<string, unknown> | null;
};

type Pipeline = {
  id: string;
  work_name: string;
  concessionaria: string;
  tipo: string;
  status: string;
  overall_score: number;
  summary_output: { executive_summary?: string };
  error_message: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  agents: Agent[];
};

const AGENT_LABEL: Record<string, string> = {
  kmz_analyzer: "01 · KMZ Analyzer",
  utm_converter: "06 · UTM Converter",
  adherence_tester: "08 · Adherence Tester",
  description_filler: "02 · Description Filler",
  report_generator: "03 · Report Generator",
  anti_reprova: "04 · Anti-Reprova",
  pipeline_supervisor: "05 · Pipeline Supervisor",
  rpa_screen_learner: "07 · RPA Screen Learner",
  master_supervisor: "09 · Master Supervisor",
};

const ORDER = [
  "kmz_analyzer", "utm_converter", "adherence_tester",
  "description_filler", "report_generator", "anti_reprova",
  "pipeline_supervisor", "rpa_screen_learner", "master_supervisor",
];

const STATUS_DOT: Record<string, string> = {
  pending: "bg-slate-700 border-slate-600",
  running: "bg-cyan-500/30 border-cyan-400 animate-pulse",
  completed: "bg-emerald-500/40 border-emerald-400",
  needs_human: "bg-amber-500/40 border-amber-400",
  failed: "bg-red-500/40 border-red-400",
  cancelled: "bg-slate-600/40 border-slate-500",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "Aguardando", running: "Em execução", completed: "Concluído",
  needs_human: "Revisão humana", failed: "Falhou", cancelled: "Cancelado",
};

const PIPE_STATUS_BADGE: Record<string, string> = {
  pending: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  running: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 animate-pulse",
  completed: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  needs_human: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  failed: "bg-red-500/20 text-red-300 border-red-500/40",
};

export default function Timeline({ initial }: { initial: Pipeline }) {
  const [pipe, setPipe] = useState<Pipeline>(initial);
  const isLive = pipe.status === "pending" || pipe.status === "running";

  useEffect(() => {
    if (!isLive) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch(`/api/pipelines/${pipe.id}`, { cache: "no-store" });
        if (r.ok && !cancelled) {
          const data = await r.json();
          setPipe(data);
        }
      } catch {}
      if (!cancelled) setTimeout(tick, 1500);
    };
    setTimeout(tick, 1500);
    return () => { cancelled = true; };
  }, [pipe.id, isLive]);

  const ordered = ORDER.map((c) => pipe.agents.find((a) => a.agent_code === c)).filter(Boolean) as Agent[];
  const done = ordered.filter((a) => ["completed", "needs_human", "failed"].includes(a.status)).length;
  const pct = Math.round((done / Math.max(1, ordered.length)) * 100);

  // First completed kmz_analyzer run lets us link to the photo viewer
  const kmzRun = pipe.agents.find((a) => a.agent_code === "kmz_analyzer");

  return (
    <div className="space-y-6">
      <div>
        <Link href="/dashboard" className="text-sm text-slate-400 hover:text-accent">← Dashboard</Link>
        <div className="mt-3 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-2xl font-bold">{pipe.work_name}</h1>
            <div className="mt-1 text-sm text-slate-400 flex items-center gap-3 flex-wrap">
              <span>{pipe.concessionaria || "—"}</span>
              <span>·</span>
              <span className="uppercase">{pipe.tipo || "—"}</span>
              <span>·</span>
              <span className="font-mono text-xs">{pipe.id}</span>
            </div>
          </div>
          <span className={`text-xs uppercase font-semibold px-3 py-1.5 rounded-md border ${PIPE_STATUS_BADGE[pipe.status]}`}>
            {STATUS_LABEL[pipe.status] || pipe.status}
          </span>
        </div>
      </div>

      <div className="grid lg:grid-cols-4 gap-4">
        <div className="bg-bg-card border border-grid-line rounded-xl p-5">
          <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Progresso</div>
          <div className="mt-2 text-3xl font-bold">{pct}<span className="text-lg text-slate-500">%</span></div>
          <div className="mt-3 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
            <div className="h-full bg-accent transition-all duration-500" style={{ width: `${pct}%` }} />
          </div>
          <div className="mt-2 text-xs text-slate-500">{done} de {ordered.length} agentes</div>
        </div>
        <div className="bg-bg-card border border-grid-line rounded-xl p-5">
          <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Score geral</div>
          <div className={`mt-2 text-3xl font-bold ${
            pipe.overall_score >= 75 ? "text-emerald-400" :
            pipe.overall_score >= 50 ? "text-amber-400" : "text-slate-500"
          }`}>
            {pipe.overall_score > 0 ? pipe.overall_score.toFixed(0) : "—"}
            {pipe.overall_score > 0 && <span className="text-lg text-slate-500">/100</span>}
          </div>
        </div>
        <div className="bg-bg-card border border-grid-line rounded-xl p-5 lg:col-span-2">
          <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Resumo executivo</div>
          <p className="mt-2 text-sm text-slate-200 leading-relaxed">
            {pipe.summary_output?.executive_summary || (isLive ? "Aguardando conclusão dos agentes…" : "Sem resumo disponível.")}
          </p>
        </div>
      </div>

      {kmzRun?.run_id && kmzRun.status !== "pending" && (
        <div className="flex items-center justify-between flex-wrap gap-3">
          <Link href={`/works/${kmzRun.run_id}`} className="text-sm text-accent hover:underline">
            Ver fotos analisadas (Agente 01) →
          </Link>
          {["completed", "needs_human"].includes(pipe.status) && (
            <PipelineDownloadButtons runId={kmzRun.run_id} />
          )}
        </div>
      )}

      <section>
        <h2 className="text-lg font-semibold mb-4">Pipeline</h2>
        <div className="bg-bg-card border border-grid-line rounded-xl p-6 space-y-3">
          {ordered.map((a, i) => (
            <AgentRow key={a.agent_code} agent={a} index={i} total={ordered.length} />
          ))}
        </div>
      </section>
    </div>
  );
}

function AgentRow({ agent, index, total }: { agent: Agent; index: number; total: number }) {
  const [expanded, setExpanded] = useState(false);
  const elapsed = agent.started_at && agent.finished_at
    ? Math.round((new Date(agent.finished_at).getTime() - new Date(agent.started_at).getTime()) / 100) / 10
    : null;
  const summary = agent.output_summary || {};
  const hasSummary = agent.status !== "pending" && (
    summary.image_count != null || summary.quality_score != null ||
    summary.filled_count != null || summary.blocking_issues != null ||
    summary.ready_to_send != null || summary.decision || summary.overall_score != null ||
    agent.error
  );

  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className={`w-4 h-4 rounded-full border-2 ${STATUS_DOT[agent.status]}`} />
        {index < total - 1 && (
          <div className={`flex-1 w-px mt-1 ${
            agent.status === "completed" || agent.status === "needs_human"
              ? "bg-grid-line" : "bg-bg-elevated"
          }`} style={{ minHeight: 32 }} />
        )}
      </div>
      <div className="flex-1 pb-4">
        {/* Cabeçalho da linha */}
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="font-semibold text-white">{AGENT_LABEL[agent.agent_code]}</span>
            <span className={`text-[10px] uppercase font-semibold px-2 py-0.5 rounded border ${
              agent.status === "completed" ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" :
              agent.status === "running"   ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40" :
              agent.status === "needs_human" ? "bg-amber-500/20 text-amber-300 border-amber-500/40" :
              agent.status === "failed"   ? "bg-red-500/20 text-red-300 border-red-500/40" :
              "bg-slate-600/20 text-slate-400 border-slate-600/40"
            }`}>{STATUS_LABEL[agent.status]}</span>
            {agent.confidence_score > 0 && (
              <span className="text-xs text-slate-500 font-mono">
                conf {Math.round(agent.confidence_score * 100)}%
              </span>
            )}
            {elapsed !== null && <span className="text-xs text-slate-500">{elapsed}s</span>}
          </div>
          <div className="flex items-center gap-3">
            {agent.depends_on.length > 0 && (
              <span className="text-xs text-slate-500 font-mono">
                após: {agent.depends_on.map((d) => d.split("_")[0]).join(", ")}
              </span>
            )}
            {hasSummary && (
              <button
                onClick={() => setExpanded(v => !v)}
                className="text-xs text-slate-500 hover:text-accent transition px-2 py-0.5 rounded border border-slate-700 hover:border-accent/40"
              >
                {expanded ? "▲ ocultar" : "▼ detalhes"}
              </button>
            )}
          </div>
        </div>

        {/* Resumo sempre visível */}
        <div className="mt-1.5 text-xs text-slate-400 space-y-0.5">
          {summary.image_count != null && (
            <div>📷 {String(summary.image_count)} fotos analisadas</div>
          )}
          {summary.quality_score != null && (
            <div>⭐ qualidade {String(summary.quality_score)}/100</div>
          )}
          {summary.filled_count != null && (
            <div>✏️ {String(summary.filled_count)} descrições preenchidas</div>
          )}
          {summary.blocking_issues != null && (
            <div className={Number(summary.blocking_issues) > 0 ? "text-red-400" : ""}>
              🚫 {String(summary.blocking_issues)} bloqueios · ⚠️ {String(summary.warnings ?? 0)} avisos
            </div>
          )}
          {summary.ready_to_send != null && (
            <div className={summary.ready_to_send ? "text-emerald-400" : "text-amber-400"}>
              {summary.ready_to_send ? "✓ pronto para envio" : "⏸ requer revisão"}
            </div>
          )}
          {summary.decision && <div>🧭 Decisão: {String(summary.decision)}</div>}
          {summary.overall_score != null && (
            <div>🏁 Score geral: {String(summary.overall_score)}/100</div>
          )}
          {agent.error && agent.status === "failed" && (
            <div className="text-red-400">⚠ {agent.error}</div>
          )}
          {agent.error && agent.status !== "failed" && agent.status !== "pending" && (
            <div className="text-slate-500 italic">{agent.error}</div>
          )}
        </div>

        {/* Painel de detalhes expandível */}
        {expanded && hasSummary && (
          <AgentDetail agent={agent} />
        )}
      </div>
    </div>
  );
}

function AgentDetail({ agent }: { agent: Agent }) {
  const out = agent.output_summary as Record<string, unknown> || {};

  const DETAIL_LABELS: Record<string, Record<string, string>> = {
    kmz_analyzer: {
      image_count:         "📷 Fotos analisadas",
      quality_score:       "⭐ Score de qualidade",
      photos_progress:     "🔄 Progresso de fotos",
      placemark_count:     "📍 Placemarks (postes)",
      total_conformes:     "✅ Conformes",
      total_divergentes:   "❌ Divergentes",
      total_com_norma:     "📐 Comparados com norma",
    },
    utm_converter: {
      pontos_convertidos:  "📍 Pontos convertidos para UTM",
      sistema:             "🗺️ Sistema de coordenadas",
      zona:                "🌐 Zona UTM",
    },
    adherence_tester: {
      blocking_issues:     "🚫 Bloqueios",
      warnings:            "⚠️ Avisos",
      divergencias:        "📋 Divergências encontradas",
    },
    description_filler: {
      filled_count:        "✏️ Descrições preenchidas",
      total:               "📦 Total de estruturas",
    },
    report_generator: {
      docx_size:           "📄 Tamanho do DOCX",
      pdf_size:            "📕 Tamanho do PDF",
      total_estruturas:    "🏗️ Estruturas no relatório",
    },
    anti_reprova: {
      blocking_issues:     "🚫 Bloqueios críticos",
      warnings:            "⚠️ Avisos",
      ready_to_send:       "📤 Pronto para envio",
    },
    pipeline_supervisor: {
      decision:            "🧭 Decisão do supervisor",
      overall_score:       "🏁 Score geral",
    },
    master_supervisor: {
      overall_score:       "🏁 Score final",
      decision:            "🧭 Decisão final",
      executive_summary:   "📝 Resumo executivo",
    },
    rpa_screen_learner: {
      telas_aprendidas:    "🖥️ Telas mapeadas",
    },
  };

  const labels = DETAIL_LABELS[agent.agent_code] || {};
  const entries = Object.entries(out).filter(([k]) => k in labels);

  // Para kmz_analyzer, mostra também coordenadas de algumas estruturas
  const isKmz = agent.agent_code === "kmz_analyzer";

  return (
    <div className="mt-3 rounded-lg bg-bg-elevated border border-grid-line p-4 space-y-2">
      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
        Detalhes do agente
      </div>

      {entries.length === 0 && (
        <div className="text-xs text-slate-500">Sem detalhes disponíveis ainda.</div>
      )}

      {entries.map(([key, value]) => {
        const label = labels[key] || key;
        if (key === "executive_summary") {
          return (
            <div key={key} className="space-y-1">
              <div className="text-xs text-slate-400 font-medium">{label}</div>
              <div className="text-xs text-slate-300 leading-relaxed italic">
                {String(value)}
              </div>
            </div>
          );
        }
        if (key === "photos_progress" && typeof value === "object" && value !== null) {
          const p = value as { current: number; total: number };
          const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
          return (
            <div key={key} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">{label}</span>
                <span className="text-accent font-mono">{p.current}/{p.total} ({pct}%)</span>
              </div>
              <div className="h-1.5 bg-bg-card rounded-full overflow-hidden">
                <div className="h-full bg-accent rounded-full transition-all" style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        }
        if (key === "ready_to_send") {
          return (
            <div key={key} className="flex justify-between text-xs">
              <span className="text-slate-400">{label}</span>
              <span className={value ? "text-emerald-400" : "text-amber-400"}>
                {value ? "Sim ✓" : "Não — requer revisão"}
              </span>
            </div>
          );
        }
        return (
          <div key={key} className="flex justify-between text-xs gap-4">
            <span className="text-slate-400">{label}</span>
            <span className="text-slate-200 font-mono text-right">
              {typeof value === "boolean" ? (value ? "Sim" : "Não") : String(value ?? "—")}
            </span>
          </div>
        );
      })}

      {/* Para kmz_analyzer: mostra horários de início/fim */}
      {isKmz && agent.started_at && (
        <div className="border-t border-grid-line pt-2 mt-2 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">🕐 Início</span>
            <span className="text-slate-300 font-mono">
              {new Date(agent.started_at).toLocaleTimeString("pt-BR")}
            </span>
          </div>
          {agent.finished_at && (
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">🕑 Fim</span>
              <span className="text-slate-300 font-mono">
                {new Date(agent.finished_at).toLocaleTimeString("pt-BR")}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
