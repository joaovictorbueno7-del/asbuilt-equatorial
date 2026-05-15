import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import LearningForm from "./LearningForm";

export default async function TreinarPage() {
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-5xl mx-auto px-6 py-8 space-y-8">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Treinar Agente</h2>
          <p className="text-slate-400 text-sm">
            Envie fotos reais de campo com os rótulos corretos para ensinar o agente de visão.
            Cada caso salvo é usado automaticamente como exemplo no próximo processamento.
          </p>
        </div>
        <LearningForm />
      </main>
    </div>
  );
}
