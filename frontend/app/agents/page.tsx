import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Link from "next/link";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";

const MODE_BADGE: Record<string, string> = {
  training: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  shadow: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
  production: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
};

const MODE_DESC: Record<string, string> = {
  training: "Coletando exemplos (< 10 casos)",
  shadow: "Shadow mode (acerto < 95%)",
  production: "Produção autônoma (≥ 95%)",
};

export default async function AgentsList() {
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const r = await backendAuthed("/dashboard/stats");
  const stats = r.ok ? await r.json() : { per_agent: [] };
  const agents = stats.per_agent || [];

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        <div>
          <Link href="/dashboard" className="text-sm text-slate-400 hover:text-accent">← Dashboard</Link>
          <h1 className="mt-3 text-2xl font-bold">Agentes</h1>
          <p className="text-slate-400 text-sm mt-1">Status, score, modo e histórico de cada agente.</p>
        </div>

        <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-bg-elevated/50 border-b border-grid-line">
              <tr className="text-xs uppercase tracking-wider text-slate-400">
                <th className="text-left px-4 py-3 font-medium">Agente</th>
                <th className="text-left px-4 py-3 font-medium">Modo</th>
                <th className="text-right px-4 py-3 font-medium">Confiança</th>
                <th className="text-right px-4 py-3 font-medium">Acerto</th>
                <th className="text-right px-4 py-3 font-medium">Runs</th>
                <th className="text-right px-4 py-3 font-medium">Casos</th>
                <th className="text-right px-4 py-3 font-medium">Pendentes</th>
                <th className="text-right px-4 py-3 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {agents.map((a: {
                code: string; name: string; mode: string;
                avg_confidence: number; accuracy: number;
                runs_count: number; cases_total: number; cases_pending_feedback: number;
                is_alert: boolean;
              }) => (
                <tr key={a.code} className={`border-b border-grid-line/40 last:border-0 hover:bg-bg-elevated/30 ${a.is_alert ? "bg-amber-950/10" : ""}`}>
                  <td className="px-4 py-3">
                    <div className="font-medium text-white">{a.name}</div>
                    <code className="text-[10px] text-accent font-mono">{a.code}</code>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`text-[10px] uppercase font-semibold px-2 py-0.5 rounded border ${MODE_BADGE[a.mode]}`}>
                      {a.mode}
                    </span>
                    <div className="text-[10px] text-slate-500 mt-0.5">{MODE_DESC[a.mode]}</div>
                  </td>
                  <td className="px-4 py-3 text-right font-mono">
                    {a.avg_confidence > 0 ? (
                      <span className={a.avg_confidence >= 0.7 ? "text-emerald-400" : "text-amber-400"}>
                        {Math.round(a.avg_confidence * 100)}%
                      </span>
                    ) : <span className="text-slate-600">—</span>}
                  </td>
                  <td className="px-4 py-3 text-right font-mono">
                    {a.accuracy > 0 ? `${Math.round(a.accuracy * 100)}%` : <span className="text-slate-600">—</span>}
                  </td>
                  <td className="px-4 py-3 text-right font-mono">{a.runs_count}</td>
                  <td className="px-4 py-3 text-right font-mono">{a.cases_total}</td>
                  <td className="px-4 py-3 text-right font-mono">
                    {a.cases_pending_feedback > 0
                      ? <span className="text-amber-400">{a.cases_pending_feedback}</span>
                      : <span className="text-slate-600">0</span>}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Link href={`/agents/${a.code}`} className="text-accent hover:underline text-sm">Abrir →</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
