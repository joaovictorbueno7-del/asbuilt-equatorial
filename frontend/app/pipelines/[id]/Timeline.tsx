"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

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
        <div className="text-right">
          <Link href={`/works/${kmzRun.run_id}`} className="text-sm text-accent hover:underline">
            Ver fotos analisadas (Agente 01) →
          </Link>
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
  const elapsed = agent.started_at && agent.finished_at
    ? Math.round((new Date(agent.finished_at).getTime() - new Date(agent.started_at).getTime()) / 100) / 10
    : null;
  const summary = agent.output_summary || {};

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
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-white">{AGENT_LABEL[agent.agent_code]}</span>
            <span className={`text-[10px] uppercase font-semibold px-2 py-0.5 rounded border ${
              agent.status === "completed" ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" :
              agent.status === "running" ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40" :
              agent.status === "needs_human" ? "bg-amber-500/20 text-amber-300 border-amber-500/40" :
              agent.status === "failed" ? "bg-red-500/20 text-red-300 border-red-500/40" :
              "bg-slate-600/20 text-slate-400 border-slate-600/40"
            }`}>
              {STATUS_LABEL[agent.status]}
            </span>
            {agent.confidence_score > 0 && (
              <span className="text-xs text-slate-500 font-mono">
                conf {Math.round(agent.confidence_score * 100)}%
              </span>
            )}
            {elapsed !== null && (
              <span className="text-xs text-slate-500">{elapsed}s</span>
            )}
          </div>
          {agent.depends_on.length > 0 && (
            <span className="text-xs text-slate-500 font-mono">
              após: {agent.depends_on.map((d) => d.split("_")[0]).join(", ")}
            </span>
          )}
        </div>
        {(agent.error || summary) && (
          <div className="mt-2 text-xs text-slate-400 space-y-0.5">
            {summary.image_count !== undefined && summary.image_count !== null && (
              <div>📷 {String(summary.image_count)} fotos analisadas</div>
            )}
            {summary.quality_score !== undefined && summary.quality_score !== null && (
              <div>⭐ qualidade {String(summary.quality_score)}/100</div>
            )}
            {summary.filled_count !== undefined && summary.filled_count !== null && (
              <div>✏️ {String(summary.filled_count)} descrições completadas</div>
            )}
            {summary.blocking_issues !== undefined && summary.blocking_issues !== null && (
              <div className={Number(summary.blocking_issues) > 0 ? "text-red-400" : ""}>
                🚫 {String(summary.blocking_issues)} bloqueios · ⚠️ {String(summary.warnings ?? 0)} avisos
              </div>
            )}
            {summary.ready_to_send !== undefined && summary.ready_to_send !== null && (
              <div className={summary.ready_to_send ? "text-emerald-400" : "text-amber-400"}>
                {summary.ready_to_send ? "✓ pronto para envio" : "⏸ requer revisão"}
              </div>
            )}
            {summary.decision && <div>🧭 decisão: {String(summary.decision)}</div>}
            {summary.overall_score !== undefined && summary.overall_score !== null && (
              <div>🏁 score geral: {String(summary.overall_score)}/100</div>
            )}
            {agent.error && agent.status === "failed" && (
              <div className="text-red-400">⚠ {agent.error}</div>
            )}
            {agent.error && agent.status !== "failed" && agent.status !== "pending" && (
              <div className="text-slate-500 italic">{agent.error}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
