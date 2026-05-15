import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import TestForm from "./TestForm";

export default async function TestarPage() {
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");
  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Testar Reconhecimento</h2>
          <p className="text-slate-400 text-sm">
            Cole uma foto de campo (Ctrl+V) e veja se o agente consegue identificar
            o que é — usando os casos que você já treinou.
          </p>
        </div>
        <TestForm />
      </main>
    </div>
  );
}
