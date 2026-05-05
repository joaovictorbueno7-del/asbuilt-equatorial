import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Link from "next/link";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import TrainView from "./TrainView";

export default async function TrainPage() {
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const [statsRes, casesRes] = await Promise.all([
    backendAuthed("/agents/kmz_analyzer/stats"),
    backendAuthed("/agents/kmz_analyzer/cases?limit=500"),
  ]);
  const stats = statsRes.ok ? await statsRes.json() : null;
  const cases = casesRes.ok ? await casesRes.json() : [];

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        <div>
          <Link href="/agents/kmz_analyzer" className="text-sm text-slate-400 hover:text-accent">← Detalhe do agente</Link>
          <h1 className="mt-3 text-2xl font-bold">Treinamento — KMZ Analyzer</h1>
          <p className="text-slate-400 text-sm mt-1">
            Revise cada estrutura, corrija quando necessário. Cada feedback aprovado vira exemplo para os próximos KMZ.
          </p>
        </div>
        <TrainView initialStats={stats} initialCases={Array.isArray(cases) ? cases : []} />
      </main>
    </div>
  );
}
