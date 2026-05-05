import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { backendAuthed } from "@/lib/api";
import Header from "@/components/Header";
import Timeline from "./Timeline";

export default async function PipelinePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const c = await cookies();
  if (!c.get("access_token")) redirect("/login");

  const meRes = await backendAuthed("/users/me");
  if (!meRes.ok) redirect("/login");
  const user = await meRes.json();

  const r = await backendAuthed(`/pipelines/${id}`);
  if (!r.ok) {
    return (
      <div className="min-h-screen">
        <Header user={user} />
        <main className="max-w-3xl mx-auto px-6 py-16 text-center text-slate-400">
          Pipeline não encontrado.
        </main>
      </div>
    );
  }
  const pipeline = await r.json();

  return (
    <div className="min-h-screen">
      <Header user={user} />
      <main className="max-w-7xl mx-auto px-6 py-8">
        <Timeline initial={pipeline} />
      </main>
    </div>
  );
}
