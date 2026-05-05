import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Link from "next/link";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import Sparkline from "@/components/Sparkline";

type Stats = {
  totals: { pipelines: number; completed: number; needs_human: number; failed: number; running: number };
  success_rate: number;
  weekly_scores: { week: string; avg_score: number | null; count: number }[];
  per_agent: AgentCard[];
  alerts: { type: string; agent_code: string; agent_name: string; value: number; message: string }[];
};

type AgentCard = {
  code: string; name: string;
  avg_confidence: number;
  runs_count: number;
  completed_runs: number;
  needs_human_runs: number;
  failed_runs: number;
  cases_total: number;
  cases_pending_feedback: number;
  accuracy: number;
  mode: string;
  is_alert: boolean;
};

type PipelineRow = {
  id: string; status: string; overall_score: number;
  work_name: string; concessionaria: string; tipo: string;
  created_at: string; finished_at: string | null;
};

const STATUS_BADGE: Record<string, string> = {
  pending: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  running: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 animate-pulse",
  completed: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  needs_human: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  failed: "bg-red-500/20 text-red-300 border-red-500/40",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "Aguardando", running: "Processando", completed: "Concluída",
  needs_human: "Revisão", failed: "Falhou",
};

const MODE_BADGE: Record<string, string> = {
  training: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  shadow: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
  production: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
};

export default async function Dashboard() {
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const [statsRes, pipesRes] = await Promise.all([
    backendAuthed("/dashboard/stats"),
    backendAuthed("/pipelines?limit=10"),
  ]);
  const stats: Stats = statsRes.ok ? await statsRes.json() : {
    totals: { pipelines: 0, completed: 0, needs_human: 0, failed: 0, running: 0 },
    success_rate: 0, weekly_scores: [], per_agent: [], alerts: [],
  };
  const pipes: PipelineRow[] = pipesRes.ok ? await pipesRes.json() : [];

  return (
    <div className="min-h-screen">
      <Header user={user} />

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        <section className="bg-gradient-to-br from-bg-card to-bg-elevated border border-grid-line rounded-2xl p-6 shadow-xl">
          <div className="flex items-start justify-between gap-6 flex-wrap">
            <div>
              <h2 className="text-2xl font-bold mb-1">
                Bem-vindo, <span className="text-accent">{(user.full_name || user.email).split(" ")[0] || "operador"}</span>
              </h2>
              <p className="text-slate-400 text-sm">Pipeline de 9 agentes. Análise visual de KMZ com IA.</p>
            </div>
            <Link
              href="/works/new"
              className="inline-flex items-center gap-2 bg-accent hover:bg-accent-hover text-bg font-bold px-6 py-3.5 rounded-xl shadow-lg shadow-accent/20 hover:shadow-accent/40 transition group"
            >
              <span className="text-xl">+</span>
              <span>Nova Análise de Obra</span>
              <span className="opacity-60 group-hover:translate-x-1 transition">→</span>
            </Link>
          </div>
        </section>

        {stats.alerts.length > 0 && (
          <section className="bg-amber-950/30 border border-amber-900/50 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-amber-400">⚠</span>
              <h3 className="font-semibold text-amber-200">Atenção: {stats.alerts.length} agente(s) abaixo de 70% de confiança</h3>
            </div>
            <ul className="space-y-1">
              {stats.alerts.map((a) => (
                <li key={a.agent_code} className="text-sm text-amber-100/80 flex items-center justify-between">
                  <span>{a.message}</span>
                  <Link href={`/agents/${a.agent_code}`} className="text-amber-300 hover:underline text-xs">Revisar →</Link>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="grid lg:grid-cols-4 gap-4">
          <Stat label="Obras processadas" value={stats.totals.pipelines} />
          <Stat label="Taxa de sucesso" value={`${stats.success_rate}%`} color={stats.success_rate >= 80 ? "emerald" : stats.success_rate >= 50 ? "amber" : "red"} />
          <Stat label="Em execução" value={stats.totals.running} pulse={stats.totals.running > 0} />
          <Stat label="Precisam revisão" value={stats.totals.needs_human} color={stats.totals.needs_human > 0 ? "amber" : undefined} />
        </section>

        <section className="grid lg:grid-cols-3 gap-4">
          <div className="bg-bg-card border border-grid-line rounded-xl p-5 lg:col-span-2">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold">Evolução do score (últimas 8 semanas)</h3>
              <span className="text-xs text-slate-500">média semanal</span>
            </div>
            <Sparkline data={stats.weekly_scores} height={100} />
            <div className="mt-3 grid grid-cols-8 gap-1 text-[10px] text-slate-600 text-center">
              {stats.weekly_scores.map((w) => (
                <div key={w.week} title={`${w.week}: ${w.count} obras`}>
                  {w.avg_score !== null ? Math.round(w.avg_score) : "—"}
                </div>
              ))}
            </div>
          </div>
          <div className="bg-bg-card border border-grid-line rounded-xl p-5">
            <h3 className="font-semibold mb-3">Distribuição</h3>
            <DistribBar label="Concluídas" count={stats.totals.completed} total={stats.totals.pipelines} color="emerald" />
            <DistribBar label="Revisão" count={stats.totals.needs_human} total={stats.totals.pipelines} color="amber" />
            <DistribBar label="Falhou" count={stats.totals.failed} total={stats.totals.pipelines} color="red" />
            <DistribBar label="Em execução" count={stats.totals.running} total={stats.totals.pipelines} color="cyan" />
          </div>
        </section>

        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Status dos 9 agentes</h2>
            <Link href="/agents" className="text-sm text-accent hover:underline">Ver detalhes →</Link>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {stats.per_agent.map((a) => <AgentCardItem key={a.code} a={a} />)}
          </div>
        </section>

        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Últimas obras</h2>
          </div>
          {pipes.length === 0 ? (
            <div className="bg-bg-card border border-grid-line border-dashed rounded-xl p-12 text-center">
              <div className="text-5xl mb-3 opacity-40">📄</div>
              <p className="text-slate-400">Nenhuma obra processada ainda.</p>
              <Link href="/works/new" className="inline-block mt-4 text-accent hover:underline">
                Iniciar primeira análise →
              </Link>
            </div>
          ) : (
            <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-bg-elevated/50 border-b border-grid-line">
                  <tr className="text-xs uppercase tracking-wider text-slate-400">
                    <th className="text-left px-4 py-3 font-medium">Obra</th>
                    <th className="text-left px-4 py-3 font-medium">Concessionária</th>
                    <th className="text-left px-4 py-3 font-medium">Status</th>
                    <th className="text-right px-4 py-3 font-medium">Score</th>
                    <th className="text-right px-4 py-3 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {pipes.map((p) => (
                    <tr key={p.id} className="border-b border-grid-line/50 last:border-0 hover:bg-bg-elevated/30">
                      <td className="px-4 py-3 font-medium text-white max-w-[260px] truncate" title={p.work_name}>{p.work_name}</td>
                      <td className="px-4 py-3 text-slate-400">{p.concessionaria || "—"}</td>
                      <td className="px-4 py-3">
                        <span className={`text-[10px] uppercase font-semibold px-2 py-0.5 rounded border ${STATUS_BADGE[p.status]}`}>
                          {STATUS_LABEL[p.status] || p.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {p.overall_score > 0 ? <span className={qualityColor(p.overall_score)}>{Math.round(p.overall_score)}</span> : "—"}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Link href={`/pipelines/${p.id}`} className="text-accent hover:underline text-sm">Abrir →</Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function qualityColor(q: number) {
  if (q >= 80) return "text-emerald-400";
  if (q >= 50) return "text-amber-400";
  return "text-red-400";
}

function Stat({ label, value, color, pulse }: { label: string; value: number | string; color?: "emerald" | "amber" | "red" | "cyan"; pulse?: boolean }) {
  const c = color === "emerald" ? "text-emerald-400" : color === "amber" ? "text-amber-400" : color === "red" ? "text-red-400" : color === "cyan" ? "text-cyan-400" : "text-white";
  return (
    <div className={`bg-bg-card border border-grid-line rounded-xl p-5 ${pulse ? "animate-pulse" : ""}`}>
      <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">{label}</div>
      <div className={`mt-2 text-3xl font-bold ${c}`}>{value}</div>
    </div>
  );
}

function DistribBar({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  const fill: Record<string, string> = {
    emerald: "bg-emerald-500", amber: "bg-amber-500", red: "bg-red-500", cyan: "bg-cyan-500",
  };
  return (
    <div className="mb-2 last:mb-0">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-300">{label}</span>
        <span className="text-slate-500 font-mono">{count}</span>
      </div>
      <div className="h-1.5 bg-bg-elevated rounded-full overflow-hidden">
        <div className={`h-full ${fill[color] || "bg-slate-500"} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function AgentCardItem({ a }: { a: AgentCard }) {
  const conf = Math.round(a.avg_confidence * 100);
  const confColor = a.avg_confidence === 0 ? "text-slate-600" : a.avg_confidence >= 0.7 ? "text-emerald-400" : "text-amber-400";
  return (
    <Link
      href={`/agents/${a.code}`}
      className={`block bg-bg-card border rounded-xl p-4 hover:border-accent/50 transition ${
        a.is_alert ? "border-amber-500/40 bg-amber-950/10" : "border-grid-line"
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div>
          <div className="font-semibold text-white text-sm">{a.name}</div>
          <code className="text-[10px] text-accent font-mono">{a.code}</code>
        </div>
        <span className={`text-[9px] uppercase font-semibold px-2 py-0.5 rounded border ${MODE_BADGE[a.mode]}`}>
          {a.mode}
        </span>
      </div>
      <div className="flex items-center justify-between mt-3">
        <div>
          <div className="text-[10px] uppercase text-slate-500">Confiança</div>
          <div className={`text-xl font-bold ${confColor}`}>
            {a.avg_confidence > 0 ? `${conf}%` : "—"}
          </div>
        </div>
        <div className="text-right text-[10px] text-slate-500 space-y-0.5">
          <div>{a.runs_count} runs</div>
          <div>{a.cases_total} casos</div>
          {a.cases_pending_feedback > 0 && (
            <div className="text-amber-400">{a.cases_pending_feedback} sem feedback</div>
          )}
        </div>
      </div>
    </Link>
  );
}
