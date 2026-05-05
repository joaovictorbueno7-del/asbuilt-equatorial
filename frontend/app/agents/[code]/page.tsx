import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Link from "next/link";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import CasesList from "./CasesList";

const MODE_BADGE: Record<string, string> = {
  training: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  shadow: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
  production: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
};

const STATUS_BADGE: Record<string, string> = {
  completed: "text-emerald-400",
  needs_human: "text-amber-400",
  failed: "text-red-400",
  running: "text-cyan-400",
  pending: "text-slate-500",
};

export default async function AgentDetail({ params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const [statsRes, casesRes] = await Promise.all([
    backendAuthed(`/agents/${code}/stats`),
    backendAuthed(`/agents/${code}/cases?limit=200`),
  ]);
  if (!statsRes.ok) {
    return (
      <div className="min-h-screen">
        <Header user={user} />
        <main className="max-w-3xl mx-auto px-6 py-16 text-center text-slate-400">
          Agente não encontrado.
        </main>
      </div>
    );
  }
  const a = await statsRes.json();
  const cases = casesRes.ok ? await casesRes.json() : [];

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        <div>
          <Link href="/agents" className="text-sm text-slate-400 hover:text-accent">← Agentes</Link>
          <div className="mt-3 flex items-start justify-between gap-3 flex-wrap">
            <div>
              <h1 className="text-2xl font-bold">{a.name}</h1>
              <code className="text-xs text-accent font-mono">{a.code} · v{a.version}</code>
              <p className="mt-2 text-slate-400 text-sm max-w-3xl">{a.description}</p>
            </div>
            <div className="flex items-center gap-3">
              {code === "kmz_analyzer" && (
                <Link href="/agents/kmz_analyzer/train"
                      className="px-3 py-1.5 rounded-lg bg-accent hover:bg-accent-hover text-bg font-semibold text-sm">
                  🎓 Treinar
                </Link>
              )}
              <span className={`text-xs uppercase font-semibold px-3 py-1.5 rounded-md border ${MODE_BADGE[a.mode]}`}>
                {a.mode}
              </span>
            </div>
          </div>
        </div>

        <section className="grid lg:grid-cols-4 gap-4">
          <Stat label="Confiança média" value={a.avg_confidence > 0 ? `${Math.round(a.avg_confidence * 100)}%` : "—"} ok={a.avg_confidence >= 0.7} />
          <Stat label="Acerto humano" value={a.accuracy > 0 ? `${Math.round(a.accuracy * 100)}%` : "—"} ok={a.accuracy >= 0.95} />
          <Stat label="Total runs" value={a.totals.runs} />
          <Stat label="Pendem feedback" value={a.totals.cases_pending_feedback} alert={a.totals.cases_pending_feedback > 0} />
        </section>

        <section className="grid lg:grid-cols-3 gap-4">
          <div className="bg-bg-card border border-grid-line rounded-xl p-5">
            <h3 className="font-semibold mb-3 text-sm">Distribuição de runs</h3>
            <Row label="Concluídos" value={a.totals.completed} color="text-emerald-400" />
            <Row label="Revisão humana" value={a.totals.needs_human} color="text-amber-400" />
            <Row label="Falhou" value={a.totals.failed} color="text-red-400" />
          </div>
          <div className="bg-bg-card border border-grid-line rounded-xl p-5 lg:col-span-2">
            <h3 className="font-semibold mb-3 text-sm">Últimas execuções</h3>
            {a.recent_runs.length === 0 ? (
              <p className="text-sm text-slate-500">Sem execuções ainda.</p>
            ) : (
              <ul className="divide-y divide-grid-line/40">
                {a.recent_runs.map((r: { id: string; status: string; confidence_score: number; created_at: string; pipeline_id: string | null }) => (
                  <li key={r.id} className="py-2 flex items-center justify-between text-sm">
                    <div className="flex items-center gap-3">
                      <span className={`text-xs uppercase font-mono ${STATUS_BADGE[r.status] || "text-slate-500"}`}>{r.status}</span>
                      <span className="text-slate-500 text-xs font-mono">{r.id.slice(0, 8)}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="font-mono text-slate-400">{Math.round(r.confidence_score * 100)}%</span>
                      <span className="text-slate-600">{new Date(r.created_at).toLocaleString("pt-BR")}</span>
                      {r.pipeline_id && (
                        <Link href={`/pipelines/${r.pipeline_id}`} className="text-accent hover:underline">→</Link>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-3">Casos para feedback</h2>
          <CasesList agentCode={code} initialCases={cases} />
        </section>
      </main>
    </div>
  );
}

function Stat({ label, value, ok, alert }: { label: string; value: string | number; ok?: boolean; alert?: boolean }) {
  const color = ok ? "text-emerald-400" : alert ? "text-amber-400" : "text-white";
  return (
    <div className="bg-bg-card border border-grid-line rounded-xl p-5">
      <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">{label}</div>
      <div className={`mt-2 text-3xl font-bold ${color}`}>{value}</div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-slate-300">{label}</span>
      <span className={`font-mono font-semibold ${color}`}>{value}</span>
    </div>
  );
}
