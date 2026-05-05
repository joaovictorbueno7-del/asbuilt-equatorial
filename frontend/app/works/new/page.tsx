import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import NewWorkForm from "./NewWorkForm";

export default async function NewWork() {
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-3xl mx-auto px-6 py-8">
        <div className="mb-6">
          <a href="/dashboard" className="text-sm text-slate-400 hover:text-accent">← Voltar ao dashboard</a>
          <h1 className="mt-3 text-2xl font-bold">Nova Análise de Obra</h1>
          <p className="text-slate-400 text-sm mt-1">
            Faça upload de um arquivo KMZ. As fotos serão extraídas e analisadas com Claude Vision.
          </p>
        </div>
        <NewWorkForm />
      </main>
    </div>
  );
}
