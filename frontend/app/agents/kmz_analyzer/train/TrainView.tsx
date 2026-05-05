"use client";

import { useState, useMemo } from "react";

type Output = {
  structure_type?: string;
  condition?: string;
  details?: string;
  confidence?: number;
  non_conformities?: string[];
};

type Case = {
  id: string;
  agent_code: string;
  feedback_score: number;
  is_correct: boolean;
  human_notes: string;
  input: { image_key?: string; placemark?: string | null };
  output: Output;
  source_run_id?: string;
  created_at: string;
};

type Stats = {
  mode: string;
  avg_confidence: number;
  accuracy: number;
  totals: { runs: number; cases: number; cases_pending_feedback: number };
};

const STRUCTURE_TYPES = ["poste", "transformador", "chave", "cruzeta", "isolador", "cabo", "para_raios", "medidor", "outro"];
const CONDITIONS = ["boa", "regular", "ruim"];

export default function TrainView({ initialStats, initialCases }: {
  initialStats: Stats | null;
  initialCases: (Case & { source_run_id?: string })[];
}) {
  const [cases, setCases] = useState(initialCases);
  const [stats, setStats] = useState(initialStats);
  const [idx, setIdx] = useState(() => {
    const firstPending = initialCases.findIndex((c) => c.feedback_score === 0.5 && !c.human_notes);
    return firstPending >= 0 ? firstPending : 0;
  });
  const [busy, setBusy] = useState(false);
  const [editingDetails, setEditingDetails] = useState<string | null>(null);
  const [correctMode, setCorrectMode] = useState(false);
  const [draft, setDraft] = useState<Output>({});

  const reviewed = useMemo(() =>
    cases.filter((c) => c.feedback_score !== 0.5 || c.human_notes).length, [cases]);
  const last10 = useMemo(() => {
    const sorted = [...cases].filter((c) => c.feedback_score !== 0.5 || c.human_notes)
      .sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 10);
    if (!sorted.length) return 0;
    return Math.round(sorted.filter((c) => c.feedback_score >= 0.55 || (c.is_correct && c.feedback_score > 0.4)).length / sorted.length * 100);
  }, [cases]);

  if (cases.length === 0) {
    return (
      <div className="bg-bg-card border border-grid-line border-dashed rounded-xl p-12 text-center">
        <div className="text-5xl mb-3 opacity-40">🎓</div>
        <p className="text-slate-300 font-semibold">Nenhum caso para revisar ainda.</p>
        <p className="text-slate-500 text-sm mt-2">
          Faça upload de um KMZ em <a href="/works/new" className="text-accent hover:underline">Nova Análise</a> e volte aqui depois.
        </p>
      </div>
    );
  }

  const cur = cases[idx];
  const conf = Math.round((cur.output.confidence ?? 0) * 100);

  async function refreshStats() {
    const r = await fetch("/api/agents/kmz_analyzer/stats");
    if (r.ok) setStats(await r.json());
  }

  function next() {
    setCorrectMode(false);
    setEditingDetails(null);
    setDraft({});
    setIdx((i) => Math.min(cases.length - 1, i + 1));
  }

  async function approve() {
    setBusy(true);
    try {
      const r = await fetch(`/api/agents/kmz_analyzer/cases/${cur.id}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_correct: true, notes: "" }),
      });
      if (r.ok) {
        const data = await r.json();
        setCases((prev) => prev.map((c) => c.id === cur.id
          ? { ...c, feedback_score: data.feedback_score, is_correct: true, human_notes: "approved" }
          : c));
        await refreshStats();
        next();
      }
    } finally { setBusy(false); }
  }

  function startCorrection() {
    setCorrectMode(true);
    setDraft({
      structure_type: cur.output.structure_type,
      condition: cur.output.condition,
      details: cur.output.details,
      non_conformities: cur.output.non_conformities || [],
    });
  }

  async function saveCorrection() {
    setBusy(true);
    try {
      const r = await fetch(`/api/agents/kmz_analyzer/cases/${cur.id}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          structure_type: draft.structure_type,
          condition: draft.condition,
          details: draft.details,
          non_conformities: draft.non_conformities || [],
          notes: "corrected",
        }),
      });
      if (r.ok) {
        const data = await r.json();
        setCases((prev) => prev.map((c) => c.id === cur.id
          ? { ...c, feedback_score: data.feedback_score, is_correct: true,
              human_notes: "corrected", output: { ...c.output, ...draft } }
          : c));
        await refreshStats();
        next();
      }
    } finally { setBusy(false); }
  }

  async function saveEdit() {
    if (editingDetails === null) return;
    setBusy(true);
    try {
      const r = await fetch(`/api/agents/kmz_analyzer/cases/${cur.id}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          details: editingDetails,
          notes: "details edited",
        }),
      });
      if (r.ok) {
        const data = await r.json();
        setCases((prev) => prev.map((c) => c.id === cur.id
          ? { ...c, feedback_score: data.feedback_score, is_correct: true,
              human_notes: "edited", output: { ...c.output, details: editingDetails } }
          : c));
        await refreshStats();
        setEditingDetails(null);
        next();
      }
    } finally { setBusy(false); }
  }

  return (
    <div className="space-y-6">
      <LearningPanel stats={stats} reviewed={reviewed} total={cases.length} last10={last10} />

      <div className="flex items-center gap-3">
        <button
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          className="px-3 py-1.5 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated disabled:opacity-30"
        >
          ← Anterior
        </button>
        <div className="flex-1 text-center text-sm text-slate-400">
          Revisado <span className="font-mono text-white">{reviewed}</span> de <span className="font-mono text-white">{cases.length}</span>
          <div className="mt-1 h-1.5 max-w-md mx-auto bg-bg-elevated rounded-full overflow-hidden">
            <div className="h-full bg-accent" style={{ width: `${(reviewed / cases.length) * 100}%` }} />
          </div>
        </div>
        <button
          onClick={() => setIdx((i) => Math.min(cases.length - 1, i + 1))}
          disabled={idx >= cases.length - 1}
          className="px-3 py-1.5 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated disabled:opacity-30"
        >
          Próxima →
        </button>
      </div>

      <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden">
        <div className="grid lg:grid-cols-2 gap-0">
          <div className="bg-bg-elevated aspect-[4/3] lg:aspect-auto flex items-center justify-center">
            {cur.source_run_id && cur.input.image_key ? (
              <img
                src={`/api/works/${cur.source_run_id}/image?key=${encodeURIComponent(cur.input.image_key)}`}
                alt={cur.input.image_key}
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-slate-500 text-sm">Foto indisponível</div>
            )}
          </div>
          <div className="p-6 space-y-4">
            <div>
              <div className="text-xs uppercase text-slate-500">Caso</div>
              <code className="text-xs font-mono text-accent">{cur.id.slice(0, 8)}</code>
              {cur.input.placemark && (
                <div className="mt-1 font-semibold text-white">{cur.input.placemark}</div>
              )}
              <div className="text-xs text-slate-500 font-mono mt-0.5">{cur.input.image_key}</div>
            </div>

            <div className="border-t border-grid-line pt-4">
              <div className="text-xs uppercase text-slate-500 mb-2">O que o agente identificou</div>
              {!correctMode && editingDetails === null ? (
                <div className="space-y-2">
                  <div className="flex gap-3 text-sm">
                    <Pill label="Tipo" value={cur.output.structure_type || "—"} />
                    <Pill label="Condição" value={cur.output.condition || "—"} />
                  </div>
                  <p className="text-sm text-slate-200">{cur.output.details || "(sem detalhes)"}</p>
                  {(cur.output.non_conformities || []).length > 0 && (
                    <div>
                      <div className="text-xs text-red-400 mb-1">Não-conformidades:</div>
                      <ul className="text-xs text-red-300 space-y-0.5">
                        {cur.output.non_conformities!.map((nc, i) => <li key={i}>⚠ {nc}</li>)}
                      </ul>
                    </div>
                  )}
                  <ConfBar conf={conf} />
                </div>
              ) : null}

              {correctMode && (
                <div className="space-y-3">
                  <Field label="Tipo correto">
                    <select className="form-input"
                      value={draft.structure_type || ""}
                      onChange={(e) => setDraft({ ...draft, structure_type: e.target.value })}>
                      {STRUCTURE_TYPES.map((s) => <option key={s} value={s}>{s}</option>)}
                    </select>
                  </Field>
                  <Field label="Condição correta">
                    <select className="form-input"
                      value={draft.condition || ""}
                      onChange={(e) => setDraft({ ...draft, condition: e.target.value })}>
                      {CONDITIONS.map((s) => <option key={s} value={s}>{s}</option>)}
                    </select>
                  </Field>
                  <Field label="Não-conformidades (uma por linha)">
                    <textarea className="form-input" rows={3}
                      value={(draft.non_conformities || []).join("\n")}
                      onChange={(e) => setDraft({ ...draft, non_conformities: e.target.value.split("\n").filter(Boolean) })} />
                  </Field>
                  <Field label="Descrição correta">
                    <textarea className="form-input" rows={4}
                      value={draft.details || ""}
                      onChange={(e) => setDraft({ ...draft, details: e.target.value })} />
                  </Field>
                </div>
              )}

              {editingDetails !== null && (
                <Field label="Editar descrição">
                  <textarea className="form-input" rows={5}
                    value={editingDetails}
                    onChange={(e) => setEditingDetails(e.target.value)} />
                </Field>
              )}
            </div>

            <div className="border-t border-grid-line pt-4 flex flex-wrap gap-2 justify-end">
              {!correctMode && editingDetails === null && (
                <>
                  <button onClick={approve} disabled={busy}
                    className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold disabled:opacity-50">
                    ✓ Correto
                  </button>
                  <button onClick={startCorrection} disabled={busy}
                    className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-semibold disabled:opacity-50">
                    ✗ Errado
                  </button>
                  <button onClick={() => setEditingDetails(cur.output.details || "")} disabled={busy}
                    className="px-4 py-2 rounded-lg border border-grid-line text-slate-200 hover:bg-bg-elevated">
                    ✏️ Corrigir descrição
                  </button>
                  <button onClick={next} className="px-4 py-2 rounded-lg text-slate-400 hover:text-white">
                    Pular →
                  </button>
                </>
              )}
              {correctMode && (
                <>
                  <button onClick={() => setCorrectMode(false)} className="px-4 py-2 rounded-lg text-slate-400 hover:text-white">
                    Cancelar
                  </button>
                  <button onClick={saveCorrection} disabled={busy}
                    className="px-4 py-2 rounded-lg bg-accent text-bg font-semibold disabled:opacity-50">
                    Salvar correção
                  </button>
                </>
              )}
              {editingDetails !== null && (
                <>
                  <button onClick={() => setEditingDetails(null)} className="px-4 py-2 rounded-lg text-slate-400 hover:text-white">
                    Cancelar
                  </button>
                  <button onClick={saveEdit} disabled={busy}
                    className="px-4 py-2 rounded-lg bg-accent text-bg font-semibold disabled:opacity-50">
                    Salvar edição
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <style jsx global>{`
        .form-input {
          width: 100%;
          background-color: #0a0e1a;
          border: 1px solid #1e293b;
          color: #e5e7eb;
          padding: 0.5rem 0.75rem;
          border-radius: 0.5rem;
          outline: none;
          font-size: 0.875rem;
        }
        .form-input:focus { border-color: #22d3ee; box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15); }
      `}</style>
    </div>
  );
}

function Pill({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className="text-sm font-semibold text-white">{value}</div>
    </div>
  );
}

function ConfBar({ conf }: { conf: number }) {
  const color = conf >= 70 ? "bg-emerald-500" : conf >= 40 ? "bg-amber-500" : "bg-red-500";
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-500">Confiança do agente</span>
        <span className="font-mono text-slate-300">{conf}%</span>
      </div>
      <div className="h-1.5 bg-bg-elevated rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${conf}%` }} />
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase text-slate-500 mb-1 block">{label}</span>
      {children}
    </label>
  );
}

function LearningPanel({ stats, reviewed, total, last10 }: {
  stats: Stats | null; reviewed: number; total: number; last10: number;
}) {
  const conf = stats ? Math.round(stats.avg_confidence * 100) : 0;
  const status = conf < 70
    ? { label: "⚠️ Agente em treinamento intensivo", color: "text-amber-300", bg: "bg-amber-950/30 border-amber-900/50" }
    : conf <= 90
      ? { label: "📈 Agente evoluindo bem", color: "text-cyan-200", bg: "bg-cyan-950/30 border-cyan-900/50" }
      : { label: "✅ Agente pronto para produção", color: "text-emerald-200", bg: "bg-emerald-950/30 border-emerald-900/50" };

  return (
    <div className={`border rounded-xl p-5 ${status.bg}`}>
      <div className={`text-base font-semibold mb-4 ${status.color}`}>{status.label}</div>
      <div className="grid sm:grid-cols-4 gap-4">
        <Stat label="Confiança média" value={conf > 0 ? `${conf}%` : "—"} />
        <Stat label="Casos revisados" value={`${reviewed} / ${total}`} />
        <Stat label="Acerto últimas 10" value={last10 > 0 ? `${last10}%` : "—"} />
        <Stat label="Modo" value={stats?.mode || "—"} />
      </div>
      <div className="mt-4">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-slate-400">Progresso até produção (90%)</span>
          <span className="font-mono text-slate-300">{conf}/90</span>
        </div>
        <div className="h-2 bg-bg-elevated rounded-full overflow-hidden">
          <div className={`h-full transition-all ${conf >= 90 ? "bg-emerald-500" : conf >= 70 ? "bg-cyan-500" : "bg-amber-500"}`}
               style={{ width: `${Math.min(100, (conf / 90) * 100)}%` }} />
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs uppercase text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-bold text-white">{value}</div>
    </div>
  );
}
