"use client";

import { useState, useMemo } from "react";

/* ─── Tipos ─────────────────────────────────────────────────────────────── */
type NormDrawing = {
  codigo: string;
  norm_id: string;
  pagina: number;
  nome_completo: string;
  has_drawing: boolean;
};

type Analysis = {
  // Campos legados (análise básica)
  structure_type?: string;
  condition?: string;
  details?: string;
  confidence?: number;
  non_conformities?: string[];
  // Campos de comparação normativa
  conformidade?: boolean | null;
  confianca?: number;
  poste?: string;
  estruturas_declaradas?: string[];
  estruturas_confirmadas?: string[];
  estruturas_divergentes?: string[];
  materiais_visiveis?: string[];
  materiais_faltantes?: string[];
  observacoes?: string;
  norm_drawings?: NormDrawing[];
};

type Case = {
  id: string;
  agent_code: string;
  feedback_score: number;
  is_correct: boolean;
  human_notes: string;
  input: {
    image_key?: string;
    placemark?: string | null;
    estruturas_declaradas?: string[];
  };
  output: Analysis;
  source_run_id?: string;
  created_at: string;
};

type Stats = {
  mode: string;
  avg_confidence: number;
  accuracy: number;
  totals: { runs: number; cases: number; cases_pending_feedback: number };
};

const STRUCTURE_TYPES = [
  "poste", "transformador", "chave", "cruzeta", "isolador",
  "cabo", "para_raios", "medidor", "outro",
];
const CONDITIONS = ["boa", "regular", "ruim"];

/* ─── Componente principal ───────────────────────────────────────────────── */
export default function TrainView({
  initialStats,
  initialCases,
}: {
  initialStats: Stats | null;
  initialCases: Case[];
}) {
  const [cases, setCases] = useState(initialCases);
  const [stats, setStats] = useState(initialStats);
  const [idx, setIdx] = useState(() => {
    const first = initialCases.findIndex(
      (c) => c.feedback_score === 0.5 && !c.human_notes
    );
    return first >= 0 ? first : 0;
  });
  const [busy, setBusy] = useState(false);
  const [correctMode, setCorrectMode] = useState(false);
  const [draft, setDraft] = useState<Analysis>({});
  // Índice do desenho técnico sendo visualizado (para estruturas multi-código)
  const [drawingIdx, setDrawingIdx] = useState(0);

  const reviewed = useMemo(
    () => cases.filter((c) => c.feedback_score !== 0.5 || c.human_notes).length,
    [cases]
  );
  const last10Pct = useMemo(() => {
    const done = [...cases]
      .filter((c) => c.feedback_score !== 0.5 || c.human_notes)
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .slice(0, 10);
    if (!done.length) return 0;
    return Math.round(
      (done.filter((c) => c.feedback_score >= 0.55 || (c.is_correct && c.feedback_score > 0.4)).length /
        done.length) *
        100
    );
  }, [cases]);

  if (!cases.length) {
    return (
      <div className="bg-bg-card border border-grid-line border-dashed rounded-xl p-12 text-center">
        <div className="text-5xl mb-3 opacity-40">🎓</div>
        <p className="text-slate-300 font-semibold">Nenhum caso para revisar ainda.</p>
        <p className="text-slate-500 text-sm mt-2">
          Faça upload de um KMZ em{" "}
          <a href="/works/new" className="text-accent hover:underline">
            Nova Análise
          </a>{" "}
          e volte aqui depois.
        </p>
      </div>
    );
  }

  const cur = cases[idx];
  const hasNormComparison = !!(
    cur.output.estruturas_declaradas?.length ||
    cur.output.norm_drawings?.length
  );
  const drawings = cur.output.norm_drawings || [];
  const curDrawing = drawings[drawingIdx] ?? null;

  async function refreshStats() {
    const r = await fetch("/api/agents/kmz_analyzer/stats");
    if (r.ok) setStats(await r.json());
  }

  function goNext() {
    setCorrectMode(false);
    setDraft({});
    setDrawingIdx(0);
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
        setCases((prev) =>
          prev.map((c) =>
            c.id === cur.id
              ? { ...c, feedback_score: data.feedback_score, is_correct: true, human_notes: "approved" }
              : c
          )
        );
        await refreshStats();
        goNext();
      }
    } finally {
      setBusy(false);
    }
  }

  function startCorrection() {
    setCorrectMode(true);
    setDraft({
      structure_type: cur.output.structure_type,
      condition: cur.output.condition,
      details: cur.output.details,
      non_conformities: cur.output.non_conformities || [],
      // Para fluxo normativo: permite corrigir estrutura real
      estruturas_confirmadas: cur.output.estruturas_confirmadas || [],
      observacoes: cur.output.observacoes || "",
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
          estruturas_confirmadas: draft.estruturas_confirmadas || [],
          observacoes: draft.observacoes || "",
          notes: "corrected",
        }),
      });
      if (r.ok) {
        const data = await r.json();
        setCases((prev) =>
          prev.map((c) =>
            c.id === cur.id
              ? {
                  ...c,
                  feedback_score: data.feedback_score,
                  is_correct: true,
                  human_notes: "corrected",
                  output: { ...c.output, ...draft },
                }
              : c
          )
        );
        await refreshStats();
        goNext();
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <LearningPanel
        stats={stats}
        reviewed={reviewed}
        total={cases.length}
        last10={last10Pct}
      />

      {/* Navegação entre casos */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => { setDrawingIdx(0); setIdx((i) => Math.max(0, i - 1)); }}
          disabled={idx === 0}
          className="px-3 py-1.5 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated disabled:opacity-30"
        >
          ← Anterior
        </button>
        <div className="flex-1 text-center text-sm text-slate-400">
          Revisado{" "}
          <span className="font-mono text-white">{reviewed}</span> de{" "}
          <span className="font-mono text-white">{cases.length}</span>
          <div className="mt-1 h-1.5 max-w-md mx-auto bg-bg-elevated rounded-full overflow-hidden">
            <div
              className="h-full bg-accent"
              style={{ width: `${(reviewed / cases.length) * 100}%` }}
            />
          </div>
        </div>
        <button
          onClick={() => { setDrawingIdx(0); setIdx((i) => Math.min(cases.length - 1, i + 1)); }}
          disabled={idx >= cases.length - 1}
          className="px-3 py-1.5 rounded-lg border border-grid-line text-slate-300 hover:bg-bg-elevated disabled:opacity-30"
        >
          Próxima →
        </button>
      </div>

      {/* Card principal */}
      <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden">

        {/* Header do placemark */}
        {(cur.input.placemark || cur.output.poste) && (
          <div className="px-6 py-3 border-b border-grid-line bg-bg-elevated flex items-center gap-3 flex-wrap">
            <span className="text-xs uppercase text-slate-500">Placemark</span>
            <span className="font-semibold text-white">
              {cur.output.poste || cur.input.placemark}
            </span>
            {hasNormComparison && (
              <ConformidadeBadge ok={cur.output.conformidade ?? null} />
            )}
          </div>
        )}

        {/* Grid de imagens */}
        <div className={`grid gap-0 ${hasNormComparison && curDrawing?.has_drawing ? "lg:grid-cols-2" : "grid-cols-1"}`}>

          {/* Foto do campo */}
          <div className="bg-bg-elevated aspect-[4/3] flex flex-col">
            <div className="text-xs text-slate-500 px-3 py-1.5 border-b border-grid-line">
              📷 Foto do campo
              {cur.input.image_key && (
                <span className="ml-2 font-mono text-slate-600">{cur.input.image_key.split("/").pop()}</span>
              )}
            </div>
            <div className="flex-1 flex items-center justify-center">
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
          </div>

          {/* Desenho técnico da norma (somente se comparação disponível) */}
          {hasNormComparison && curDrawing?.has_drawing && (
            <div className="bg-[#0a0e1a] aspect-[4/3] flex flex-col border-l border-grid-line">
              <div className="text-xs text-slate-500 px-3 py-1.5 border-b border-grid-line flex items-center justify-between">
                <span>📐 Desenho técnico norma — <strong className="text-cyan-400">{curDrawing.codigo}</strong></span>
                {drawings.length > 1 && (
                  <div className="flex gap-1">
                    {drawings.map((d, i) => (
                      <button
                        key={d.codigo}
                        onClick={() => setDrawingIdx(i)}
                        className={`px-2 py-0.5 rounded text-xs font-mono transition-colors ${
                          i === drawingIdx
                            ? "bg-accent text-bg font-bold"
                            : "border border-grid-line text-slate-400 hover:text-white"
                        }`}
                      >
                        {d.codigo}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="flex-1 flex items-center justify-center">
                {curDrawing.norm_id && curDrawing.pagina > 0 ? (
                  <img
                    src={`/api/knowledge/${curDrawing.norm_id}/page_image?page=${curDrawing.pagina}`}
                    alt={`Desenho técnico ${curDrawing.codigo}`}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-slate-500 text-sm">Desenho não disponível</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Painel de análise */}
        <div className="p-6 space-y-4">

          {/* Análise normativa (fluxo novo) */}
          {hasNormComparison && !correctMode && (
            <NormComparisonPanel analysis={cur.output} />
          )}

          {/* Análise básica (fluxo legado — sem norma) */}
          {!hasNormComparison && !correctMode && (
            <BasicAnalysisPanel analysis={cur.output} />
          )}

          {/* Modo correção */}
          {correctMode && (
            <CorrectionForm
              draft={draft}
              setDraft={setDraft}
              hasNorm={hasNormComparison}
              declaredCodes={cur.output.estruturas_declaradas || []}
            />
          )}

          {/* Botões de ação */}
          <div className="border-t border-grid-line pt-4 flex flex-wrap gap-2 justify-end">
            {!correctMode ? (
              <>
                <button
                  onClick={approve}
                  disabled={busy}
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold disabled:opacity-50"
                >
                  ✅ Confirmado
                </button>
                <button
                  onClick={startCorrection}
                  disabled={busy}
                  className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-semibold disabled:opacity-50"
                >
                  ❌ Divergência
                </button>
                <button
                  onClick={goNext}
                  className="px-4 py-2 rounded-lg text-slate-400 hover:text-white"
                >
                  Pular →
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => setCorrectMode(false)}
                  className="px-4 py-2 rounded-lg text-slate-400 hover:text-white"
                >
                  Cancelar
                </button>
                <button
                  onClick={saveCorrection}
                  disabled={busy}
                  className="px-4 py-2 rounded-lg bg-accent text-bg font-semibold disabled:opacity-50"
                >
                  ✏️ Salvar correção
                </button>
              </>
            )}
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
        .form-input:focus {
          border-color: #22d3ee;
          box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15);
        }
      `}</style>
    </div>
  );
}

/* ─── Painel comparação normativa ────────────────────────────────────────── */
function NormComparisonPanel({ analysis }: { analysis: Analysis }) {
  const conf = Math.round((analysis.confianca ?? analysis.confidence ?? 0) * 100);
  return (
    <div className="space-y-4">
      {/* Estruturas declaradas vs confirmadas */}
      <div className="grid sm:grid-cols-2 gap-3">
        <div>
          <div className="text-xs uppercase text-slate-500 mb-1">Declarado no KMZ</div>
          <div className="flex flex-wrap gap-1">
            {(analysis.estruturas_declaradas || []).map((c) => (
              <span key={c} className="px-2 py-0.5 rounded bg-slate-700 text-slate-200 text-xs font-mono">
                {c}
              </span>
            ))}
            {!analysis.estruturas_declaradas?.length && (
              <span className="text-slate-500 text-xs">—</span>
            )}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500 mb-1">Confirmado por Vision</div>
          <div className="flex flex-wrap gap-1">
            {(analysis.estruturas_confirmadas || []).map((c) => (
              <span key={c} className="px-2 py-0.5 rounded bg-emerald-900/60 text-emerald-300 text-xs font-mono border border-emerald-800/50">
                ✓ {c}
              </span>
            ))}
            {(analysis.estruturas_divergentes || []).map((c) => (
              <span key={c} className="px-2 py-0.5 rounded bg-red-900/60 text-red-300 text-xs font-mono border border-red-800/50">
                ✗ {c}
              </span>
            ))}
            {!analysis.estruturas_confirmadas?.length && !analysis.estruturas_divergentes?.length && (
              <span className="text-slate-500 text-xs">aguardando análise</span>
            )}
          </div>
        </div>
      </div>

      {/* Materiais */}
      {(analysis.materiais_faltantes?.length ?? 0) > 0 && (
        <div>
          <div className="text-xs uppercase text-amber-500 mb-1">Materiais faltantes</div>
          <ul className="text-xs text-amber-300 space-y-0.5">
            {analysis.materiais_faltantes!.map((m, i) => (
              <li key={i}>⚠ {m}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Observações */}
      {analysis.observacoes && (
        <p className="text-sm text-slate-300 bg-bg-elevated rounded-lg px-3 py-2">
          {analysis.observacoes}
        </p>
      )}

      {/* Barra de confiança */}
      <ConfBar conf={conf} label="Confiança da análise Vision" />
    </div>
  );
}

/* ─── Painel análise básica (sem norma) ─────────────────────────────────── */
function BasicAnalysisPanel({ analysis }: { analysis: Analysis }) {
  const conf = Math.round((analysis.confidence ?? 0) * 100);
  return (
    <div className="space-y-3">
      <div className="flex gap-3 text-sm">
        <Pill label="Tipo" value={analysis.structure_type || "—"} />
        <Pill label="Condição" value={analysis.condition || "—"} />
      </div>
      <p className="text-sm text-slate-200">{analysis.details || "(sem detalhes)"}</p>
      {(analysis.non_conformities || []).length > 0 && (
        <div>
          <div className="text-xs text-red-400 mb-1">Não-conformidades:</div>
          <ul className="text-xs text-red-300 space-y-0.5">
            {analysis.non_conformities!.map((nc, i) => (
              <li key={i}>⚠ {nc}</li>
            ))}
          </ul>
        </div>
      )}
      <ConfBar conf={conf} label="Confiança do agente" />
    </div>
  );
}

/* ─── Formulário de correção ─────────────────────────────────────────────── */
function CorrectionForm({
  draft,
  setDraft,
  hasNorm,
  declaredCodes,
}: {
  draft: Analysis;
  setDraft: (d: Analysis) => void;
  hasNorm: boolean;
  declaredCodes: string[];
}) {
  return (
    <div className="space-y-4 bg-bg-elevated rounded-xl p-4 border border-amber-900/40">
      <div className="text-sm font-semibold text-amber-300">
        ✏️ Informe o que realmente está na foto — o agente aprenderá com essa correção
      </div>

      {hasNorm && (
        <>
          <Field label="Estrutura real presente (corrija o código)">
            <input
              className="form-input font-mono"
              placeholder={`Ex: N3 (declarado foi: ${declaredCodes.join(", ")})`}
              value={(draft.estruturas_confirmadas || []).join(", ")}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  estruturas_confirmadas: e.target.value
                    .split(/[\s,]+/)
                    .map((s) => s.trim().toUpperCase())
                    .filter(Boolean),
                })
              }
            />
            <span className="text-xs text-slate-500 mt-1 block">
              Use o código exato da norma (ex: N1, S3I, U3). Separe com vírgula se houver mais de uma.
            </span>
          </Field>
          <Field label="Observações da correção">
            <textarea
              className="form-input"
              rows={3}
              placeholder="Ex: A estrutura é N3, não N1 — cruzeta é simples, não dupla"
              value={draft.observacoes || ""}
              onChange={(e) => setDraft({ ...draft, observacoes: e.target.value })}
            />
          </Field>
        </>
      )}

      {!hasNorm && (
        <>
          <Field label="Tipo correto">
            <select
              className="form-input"
              value={draft.structure_type || ""}
              onChange={(e) => setDraft({ ...draft, structure_type: e.target.value })}
            >
              {STRUCTURE_TYPES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="Condição correta">
            <select
              className="form-input"
              value={draft.condition || ""}
              onChange={(e) => setDraft({ ...draft, condition: e.target.value })}
            >
              {CONDITIONS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="Não-conformidades (uma por linha)">
            <textarea
              className="form-input"
              rows={3}
              value={(draft.non_conformities || []).join("\n")}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  non_conformities: e.target.value.split("\n").filter(Boolean),
                })
              }
            />
          </Field>
          <Field label="Descrição correta">
            <textarea
              className="form-input"
              rows={4}
              value={draft.details || ""}
              onChange={(e) => setDraft({ ...draft, details: e.target.value })}
            />
          </Field>
        </>
      )}
    </div>
  );
}

/* ─── Componentes auxiliares ─────────────────────────────────────────────── */
function ConformidadeBadge({ ok }: { ok: boolean | null }) {
  if (ok === null || ok === undefined)
    return <span className="px-2 py-0.5 rounded-full text-xs bg-slate-700 text-slate-400">Sem norma</span>;
  return ok ? (
    <span className="px-2 py-0.5 rounded-full text-xs bg-emerald-900/60 text-emerald-300 border border-emerald-800/50">
      ✅ Conforme
    </span>
  ) : (
    <span className="px-2 py-0.5 rounded-full text-xs bg-red-900/60 text-red-300 border border-red-800/50">
      ❌ Divergente
    </span>
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

function ConfBar({ conf, label }: { conf: number; label: string }) {
  const color =
    conf >= 70 ? "bg-emerald-500" : conf >= 40 ? "bg-amber-500" : "bg-red-500";
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-500">{label}</span>
        <span className="font-mono text-slate-300">{conf}%</span>
      </div>
      <div className="h-1.5 bg-bg-elevated rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all`}
          style={{ width: `${conf}%` }}
        />
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

function LearningPanel({
  stats,
  reviewed,
  total,
  last10,
}: {
  stats: Stats | null;
  reviewed: number;
  total: number;
  last10: number;
}) {
  const conf = stats ? Math.round(stats.avg_confidence * 100) : 0;
  const status =
    conf < 70
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
          <div
            className={`h-full transition-all ${
              conf >= 90 ? "bg-emerald-500" : conf >= 70 ? "bg-cyan-500" : "bg-amber-500"
            }`}
            style={{ width: `${Math.min(100, (conf / 90) * 100)}%` }}
          />
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
