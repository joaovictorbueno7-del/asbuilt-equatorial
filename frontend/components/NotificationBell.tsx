"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";

type Notification = {
  id: string;
  type: string;
  level: "success" | "warning" | "error" | "milestone";
  title: string;
  message: string;
  link: string;
  ts: string;
};

const LEVEL_DOT: Record<string, string> = {
  success: "bg-emerald-400",
  warning: "bg-amber-400",
  error: "bg-red-400",
  milestone: "bg-fuchsia-400",
};

const LEVEL_BORDER: Record<string, string> = {
  success: "border-l-emerald-500",
  warning: "border-l-amber-500",
  error: "border-l-red-500",
  milestone: "border-l-fuchsia-500",
};

export default function NotificationBell() {
  const [items, setItems] = useState<Notification[]>([]);
  const [open, setOpen] = useState(false);
  const [seenIds, setSeenIds] = useState<Set<string>>(() => new Set());
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch("/api/notifications", { cache: "no-store" });
        if (r.ok && !cancelled) {
          const data = await r.json();
          setItems(data.items || []);
        }
      } catch {}
      if (!cancelled) setTimeout(tick, 8000);
    };
    tick();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (open) {
      // Mark all visible as seen on open
      setSeenIds(new Set(items.map((i) => i.id)));
    }
  }, [open, items]);

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const unread = items.filter((i) => !seenIds.has(i.id)).length;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="relative p-2 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated hover:text-white transition"
        title="Notificações"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M18 16v-5a6 6 0 1 0-12 0v5l-2 2v1h16v-1l-2-2zM10 21a2 2 0 0 0 4 0" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        {unread > 0 && (
          <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] px-1 rounded-full bg-red-500 text-[10px] font-bold flex items-center justify-center text-white border-2 border-bg">
            {unread > 9 ? "9+" : unread}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-96 max-h-[500px] overflow-auto bg-bg-card border border-grid-line rounded-xl shadow-2xl z-30">
          <div className="px-4 py-3 border-b border-grid-line flex items-center justify-between">
            <h3 className="text-sm font-semibold">Notificações</h3>
            <span className="text-xs text-slate-500">{items.length} recentes</span>
          </div>
          {items.length === 0 ? (
            <div className="p-6 text-center text-sm text-slate-500">
              Nenhuma notificação ainda.
            </div>
          ) : (
            <ul className="divide-y divide-grid-line/40">
              {items.map((n) => (
                <li key={n.id}>
                  <Link
                    href={n.link}
                    onClick={() => setOpen(false)}
                    className={`block px-4 py-3 hover:bg-bg-elevated/40 border-l-2 ${LEVEL_BORDER[n.level]}`}
                  >
                    <div className="flex items-start gap-2">
                      <span className={`mt-1.5 w-2 h-2 rounded-full ${LEVEL_DOT[n.level]} flex-shrink-0`} />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-white truncate">{n.title}</div>
                        <div className="text-xs text-slate-400 mt-0.5 line-clamp-2">{n.message}</div>
                        <div className="text-[10px] text-slate-600 mt-1">
                          {new Date(n.ts).toLocaleString("pt-BR")}
                        </div>
                      </div>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
