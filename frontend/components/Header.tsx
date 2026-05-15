import Link from "next/link";
import LogoutButton from "@/app/dashboard/LogoutButton";
import NotificationBell from "@/components/NotificationBell";

const ROLE_BADGE: Record<string, string> = {
  admin: "bg-red-500/20 text-red-300 border-red-500/30",
  supervisor: "bg-purple-500/20 text-purple-300 border-purple-500/30",
  operador: "bg-blue-500/20 text-blue-300 border-blue-500/30",
  auditor: "bg-amber-500/20 text-amber-300 border-amber-500/30",
};

export default function Header({ user }: { user: { full_name: string; email: string; role: string } }) {
  return (
    <header className="border-b border-grid-line bg-bg-card/60 backdrop-blur sticky top-0 z-20">
      <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent to-indigo-500 flex items-center justify-center font-bold text-bg group-hover:scale-105 transition">
            ⚡
          </div>
          <div>
            <h1 className="text-base font-bold leading-tight">
              <span className="text-white">OPS AI</span>{" "}
              <span className="text-accent">GRID</span>
            </h1>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Plataforma de obras</p>
          </div>
        </Link>
        <div className="flex items-center gap-4">
          <div className="text-right hidden sm:block">
            <div className="text-sm font-medium leading-tight">{user.full_name || user.email}</div>
            <div className="text-xs text-slate-500">{user.email}</div>
          </div>
          <span className={`text-[10px] uppercase font-semibold px-2 py-1 rounded-md border ${ROLE_BADGE[user.role] || "bg-slate-500/20 text-slate-300 border-slate-500/30"}`}>
            {user.role}
          </span>
          <Link
            href="/knowledge"
            className="text-sm px-3 py-1.5 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated hover:text-white transition hidden md:inline-block"
          >
            Normas
          </Link>
          <Link
            href="/agents"
            className="text-sm px-3 py-1.5 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated hover:text-white transition hidden md:inline-block"
          >
            Agentes
          </Link>
          <Link
            href="/treinar"
            className="text-sm px-3 py-1.5 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated hover:text-white transition hidden md:inline-block"
          >
            Treinar
          </Link>
          <NotificationBell />
          <LogoutButton />
        </div>
      </div>
    </header>
  );
}
