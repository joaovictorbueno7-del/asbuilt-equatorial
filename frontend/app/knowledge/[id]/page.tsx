import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Link from "next/link";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import NormActions from "./NormActions";

export default async function NormDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const r = await backendAuthed(`/knowledge/${id}`);
  if (!r.ok) {
    return (
      <div className="min-h-screen">
        <Header user={user} />
        <main className="max-w-3xl mx-auto px-6 py-16 text-center text-slate-400">Norma não encontrada.</main>
      </div>
    );
  }
  const n = await r.json();

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        <div>
          <Link href="/knowledge" className="text-sm text-slate-400 hover:text-accent">← Base de Normas</Link>
          <div className="mt-3 flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="flex items-center gap-3">
                <code className="text-base font-mono text-accent">{n.codigo}</code>
                <span className="text-xs font-mono text-slate-500">v{n.versao}</span>
                {n.ativa ? (
                  <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded border bg-emerald-500/20 text-emerald-300 border-emerald-500/40">ativa</span>
                ) : (
                  <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded border bg-slate-500/20 text-slate-400 border-slate-500/40">inativa</span>
                )}
              </div>
              <h1 className="mt-2 text-2xl font-bold">{n.nome}</h1>
              <p className="text-slate-400 text-sm mt-1">
                {n.concessionaria} · {n.page_count} páginas · {(n.pdf_size_bytes / 1024).toFixed(0)} KB · MD5 <code className="text-xs">{n.pdf_hash.slice(0,12)}</code>
              </p>
            </div>
            <NormActions normId={n.id} ativa={n.ativa} />
          </div>
        </div>

        <section className="grid lg:grid-cols-4 gap-4">
          <Stat label="Estruturas indexadas" value={n.structure_count} ok={n.structure_count > 0} />
          <Stat label="Páginas" value={n.page_count} />
          <Stat label="Vigência início" value={n.data_vigencia_inicio || "—"} />
          <Stat label="Vigência fim" value={n.data_vigencia_fim || "—"} alert={!!n.data_vigencia_fim} />
        </section>

        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Estruturas indexadas ({n.structure_count})</h2>
            <a href={`http://localhost:8000/api/v1/knowledge/${n.id}/pdf`} target="_blank"
               className="text-sm text-accent hover:underline">📄 Abrir PDF original</a>
          </div>
          {n.structure_count === 0 ? (
            <div className="bg-bg-card border border-grid-line border-dashed rounded-xl p-8 text-center text-sm text-slate-400">
              Extração em andamento. Recarregue a página em ~30 segundos.
            </div>
          ) : (
            <div className="grid md:grid-cols-2 gap-4">
              {n.structures.map((s: { id: string; codigo_estrutura: string; nome_completo: string; descricao_tecnica: string; caracteristicas_visuais: string; campos_proj: Record<string, string>; materiais: string[] }) => (
                <div key={s.id} className="bg-bg-card border border-grid-line rounded-xl p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <code className="text-sm font-mono text-accent">{s.codigo_estrutura}</code>
                    <span className="font-semibold text-white">{s.nome_completo}</span>
                  </div>
                  {s.descricao_tecnica && <p className="text-xs text-slate-300">{s.descricao_tecnica}</p>}
                  {s.caracteristicas_visuais && (
                    <div>
                      <div className="text-[10px] uppercase text-slate-500">Características visuais</div>
                      <p className="text-xs text-slate-400">{s.caracteristicas_visuais}</p>
                    </div>
                  )}
                  {Object.keys(s.campos_proj || {}).length > 0 && (
                    <div>
                      <div className="text-[10px] uppercase text-slate-500">Campos PROJ</div>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {Object.entries(s.campos_proj).map(([k, v]) => (
                          <span key={k} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-bg-elevated border border-grid-line text-slate-300">
                            {k}: {String(v)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {(s.materiais || []).length > 0 && (
                    <div>
                      <div className="text-[10px] uppercase text-slate-500">Materiais</div>
                      <ul className="text-xs text-slate-400 mt-1">
                        {s.materiais.slice(0, 6).map((m, i) => <li key={i}>· {m}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        {n.replaced_by_id && (
          <div className="bg-amber-950/20 border border-amber-900/40 rounded-xl p-4 text-sm text-amber-200">
            Esta norma foi substituída pela <Link href={`/knowledge/${n.replaced_by_id}`} className="text-amber-300 hover:underline">versão mais recente</Link>.
          </div>
        )}
      </main>
    </div>
  );
}

function Stat({ label, value, ok, alert }: { label: string; value: string | number; ok?: boolean; alert?: boolean }) {
  const color = ok ? "text-emerald-400" : alert ? "text-amber-400" : "text-white";
  return (
    <div className="bg-bg-card border border-grid-line rounded-xl p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">{label}</div>
      <div className={`mt-2 text-xl font-bold ${color}`}>{value}</div>
    </div>
  );
}
