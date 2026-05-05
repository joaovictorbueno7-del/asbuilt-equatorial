import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Link from "next/link";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import ResultView from "./ResultView";

export default async function WorkPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");

  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const workRes = await backendAuthed(`/works/${id}`);
  if (!workRes.ok) {
    return (
      <div className="min-h-screen">
        <Header user={user} />
        <main className="max-w-3xl mx-auto px-6 py-16 text-center">
          <p className="text-slate-400">Obra não encontrada.</p>
          <Link href="/dashboard" className="mt-4 inline-block text-accent hover:underline">← Voltar</Link>
        </main>
      </div>
    );
  }
  const work = await workRes.json();

  const casesRes = await backendAuthed(`/agents/kmz_analyzer/cases?source_run_id=${id}&limit=500`);
  const cases = casesRes.ok ? await casesRes.json() : [];

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-7xl mx-auto px-6 py-8">
        <ResultView work={work} cases={cases} />
      </main>
    </div>
  );
}
