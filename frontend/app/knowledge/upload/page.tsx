import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import UploadForm from "./UploadForm";

export default async function UploadNorm() {
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-3xl mx-auto px-6 py-8">
        <a href="/knowledge" className="text-sm text-slate-400 hover:text-accent">← Base de Normas</a>
        <h1 className="mt-3 text-2xl font-bold">Adicionar Norma Técnica</h1>
        <p className="text-slate-400 text-sm mt-1 mb-6">
          PDF original será preservado para sempre. Estruturas serão extraídas automaticamente com Claude.
        </p>
        <UploadForm />
      </main>
    </div>
  );
}
