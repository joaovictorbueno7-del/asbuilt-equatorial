"use client";

import { useState } from "react";
import Link from "next/link";

type Case = {
  id: string;
  agent_code: string;
  feedback_score: number;
  is_correct: boolean;
  human_notes: string;
  input: { image_key?: string; placemark?: string | null };
  output: { structure_type?: string; condition?: string; details?: string; confidence?: number; non_conformities?: string[] };
  created_at: string;
};

export default function CasesList({ agentCode, initialCases }: { agentCode: string; initialCases: Case[] }) {
  const [cases, setCases] = useState<Case[]>(Array.isArray(initialCases) ? initialCases : []);
  const [filter, setFilter] = useState<"all" | "pending" | "reviewed">("all");
  const [busy, setBusy] = useState<string | null>(null);

  const filtered = cases.filter((c) => {
    if (filter === "pending") return c.feedback_score === 0.5 && !c.human_notes;
    if (filter === "reviewed") return c.feedback_score !== 0.5 || c.human_notes;
    return true;
  });

  async function send(caseId: string, correct: boolean) {
    setBusy(caseId);
    try {
      const r = await fetch(`/api/agents/${agentCode}/cases/${caseId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_correct: correct, notes: "" }),
      });
      if (r.ok) {
        const data = await r.json();
        setCases((prev) => prev.map((c) =>
          c.id === caseId ? { ...c, feedback_score: data.feedback_score, is_correct: data.is_correct, human_notes: c.human_notes || "reviewed" } : c
        ));
      }
    } finally {
      setBusy(null);
    }
  }

  if (cases.length === 0) {
    return (
      <div className="bg-bg-card border border-grid-line border-dashed rounded-xl p-10 text-center text-slate-500 text-sm">
        Nenhum caso registrado para este agente ainda.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2 text-sm">
        {(["all", "pending", "reviewed"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-lg border transition ${
              filter === f
                ? "bg-accent text-bg border-accent font-semibold"
                : "border-grid-line text-slate-400 hover:bg-bg-elevated"
            }`}
          >
            {f === "all" ? `Todos (${cases.length})`
              : f === "pending" ? `Pendentes (${cases.filter((c) => c.feedback_score === 0.5 && !c.human_notes).length})`
              : `Revisados (${cases.filter((c) => c.feedback_score !== 0.5 || c.human_notes).length})`}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map((c) => {
          const reviewed = c.feedback_score !== 0.5 || c.human_notes;
          return (
            <div key={c.id} className="bg-bg-card border border-grid-line rounded-xl p-4 space-y-2">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <code className="text-[10px] font-mono text-accent">{c.id.slice(0, 8)}</code>
                  <div className="text-sm font-semibold text-white mt-0.5">
                    {c.input.placemark || c.input.image_key || "—"}
                  </div>
                </div>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                  c.feedback_score >= 0.55 ? "bg-emerald-500/20 text-emerald-300" :
                  c.feedback_score <= 0.45 ? "bg-red-500/20 text-red-300" :
                  "bg-slate-500/20 text-slate-400"
                }`}>
                  {c.feedback_score.toFixed(2)}
                </span>
              </div>

              {c.output.structure_type && (
                <div className="text-xs text-slate-400">
                  <span className="text-slate-500">tipo:</span> <span className="text-slate-300">{c.output.structure_type}</span>
                  {c.output.condition && <> · <span className="text-slate-300">{c.output.condition}</span></>}
                </div>
              )}
              {c.output.details && <p className="text-xs text-slate-400 line-clamp-3">{c.output.details}</p>}

              <div className="flex items-center justify-between pt-2 border-t border-grid-line/40">
                <div className="text-[10px] text-slate-600">
                  {new Date(c.created_at).toLocaleDateString("pt-BR")}
                </div>
                <div className="flex items-center gap-1.5">
                  {reviewed && (
                    <span className="text-[10px] text-slate-500">
                      {c.is_correct ? "✓ correto" : "✗ incorreto"}
                    </span>
                  )}
                  <button
                    disabled={busy === c.id}
                    onClick={() => send(c.id, true)}
                    className={`text-xs px-2 py-1 rounded border transition ${
                      reviewed && c.is_correct
                        ? "bg-emerald-500/30 border-emerald-500/60 text-emerald-200"
                        : "border-grid-line hover:border-emerald-500/50 hover:bg-emerald-500/10 text-slate-400"
                    } disabled:opacity-50`}
                  >
                    ✓
                  </button>
                  <button
                    disabled={busy === c.id}
                    onClick={() => send(c.id, false)}
                    className={`text-xs px-2 py-1 rounded border transition ${
                      reviewed && !c.is_correct
                        ? "bg-red-500/30 border-red-500/60 text-red-200"
                        : "border-grid-line hover:border-red-500/50 hover:bg-red-500/10 text-slate-400"
                    } disabled:opacity-50`}
                  >
                    ✗
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
