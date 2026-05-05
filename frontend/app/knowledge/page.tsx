import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Link from "next/link";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";

type Norm = {
  id: string; concessionaria: string; codigo: string; nome: string; versao: string;
  pdf_filename: string; pdf_size_bytes: number; page_count: number; structure_count: number;
  ativa: boolean; data_vigencia_inicio: string | null; data_vigencia_fim: string | null;
  replaced_by_id: string | null; created_at: string;
};

const CONCESSIONARIAS = ["Equatorial", "Cemig", "Copel", "Enel", "Light", "EDP", "Energisa", "Outro"];

export default async function KnowledgeList({ searchParams }: { searchParams: Promise<{ concessionaria?: string; ativa?: string }> }) {
  const sp = await searchParams;
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const qs = new URLSearchParams();
  if (sp.concessionaria) qs.set("concessionaria", sp.concessionaria);
  if (sp.ativa) qs.set("ativa", sp.ativa);
  const r = await backendAuthed(`/knowledge?${qs.toString()}`);
  const norms: Norm[] = r.ok ? await r.json() : [];

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <Link href="/dashboard" className="text-sm text-slate-400 hover:text-accent">← Dashboard</Link>
            <h1 className="mt-3 text-2xl font-bold">Base de Normas Técnicas</h1>
            <p className="text-slate-400 text-sm mt-1">
              Conhecimento permanente. <strong>Normas nunca são deletadas</strong> — apenas desativadas para preservar histórico.
            </p>
          </div>
          <Link href="/knowledge/upload"
                className="bg-accent hover:bg-accent-hover text-bg font-bold px-5 py-3 rounded-xl">
            + Adicionar Norma
          </Link>
        </div>

        <div className="flex gap-2 flex-wrap text-sm">
          <FilterPill label="Todas" href="/knowledge" active={!sp.concessionaria && !sp.ativa} />
          <FilterPill label="Ativas" href="/knowledge?ativa=true" active={sp.ativa === "true"} />
          <FilterPill label="Inativas" href="/knowledge?ativa=false" active={sp.ativa === "false"} />
          <span className="text-slate-600 mx-2">|</span>
          {CONCESSIONARIAS.map((c) => (
            <FilterPill key={c} label={c}
              href={`/knowledge?concessionaria=${encodeURIComponent(c)}`}
              active={sp.concessionaria === c} />
          ))}
        </div>

        {norms.length === 0 ? (
          <div className="bg-bg-card border border-grid-line border-dashed rounded-xl p-12 text-center">
            <div className="text-5xl mb-3 opacity-40">📚</div>
            <p className="text-slate-400">Nenhuma norma cadastrada com esses filtros.</p>
            <Link href="/knowledge/upload" className="inline-block mt-4 text-accent hover:underline">
              Adicionar primeira norma →
            </Link>
          </div>
        ) : (
          <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-bg-elevated/50 border-b border-grid-line">
                <tr className="text-xs uppercase tracking-wider text-slate-400">
                  <th className="text-left px-4 py-3 font-medium">Código</th>
                  <th className="text-left px-4 py-3 font-medium">Nome</th>
                  <th className="text-left px-4 py-3 font-medium">Concessionária</th>
                  <th className="text-left px-4 py-3 font-medium">Versão</th>
                  <th className="text-right px-4 py-3 font-medium">Páginas</th>
                  <th className="text-right px-4 py-3 font-medium">Estruturas</th>
                  <th className="text-left px-4 py-3 font-medium">Status</th>
                  <th className="text-right px-4 py-3 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {norms.map((n) => (
                  <tr key={n.id} className={`border-b border-grid-line/50 last:border-0 hover:bg-bg-elevated/30 ${!n.ativa ? "opacity-60" : ""}`}>
                    <td className="px-4 py-3 font-mono text-accent">{n.codigo}</td>
                    <td className="px-4 py-3 max-w-[300px] truncate" title={n.nome}>{n.nome}</td>
                    <td className="px-4 py-3 text-slate-400">{n.concessionaria}</td>
                    <td className="px-4 py-3 font-mono text-slate-400">v{n.versao}</td>
                    <td className="px-4 py-3 text-right font-mono">{n.page_count}</td>
                    <td className="px-4 py-3 text-right font-mono">
                      {n.structure_count > 0 ? <span className="text-emerald-400">{n.structure_count}</span> : <span className="text-slate-500">processando</span>}
                    </td>
                    <td className="px-4 py-3">
                      {n.ativa ? (
                        <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded border bg-emerald-500/20 text-emerald-300 border-emerald-500/40">ativa</span>
                      ) : (
                        <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded border bg-slate-500/20 text-slate-400 border-slate-500/40">
                          inativa{n.replaced_by_id ? " · substituída" : ""}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Link href={`/knowledge/${n.id}`} className="text-accent hover:underline text-sm">Abrir →</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}

function FilterPill({ label, href, active }: { label: string; href: string; active: boolean }) {
  return (
    <Link href={href} className={`px-3 py-1 rounded-full border text-xs ${
      active ? "bg-accent text-bg border-accent font-semibold"
             : "border-grid-line text-slate-400 hover:bg-bg-elevated"
    }`}>{label}</Link>
  );
}
