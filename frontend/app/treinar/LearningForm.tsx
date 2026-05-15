"use client";

import { useState, useRef, DragEvent, FormEvent, useEffect, useCallback, useMemo } from "react";
import AnnotationCanvas, { BBox } from "./AnnotationCanvas";

type LearningCase = {
  id: string;
  structure_codes: string[];
  pole_size: string;
  conformidade: boolean;
  notes: string;
  concessionaria: string;
  created_at: string;
};

const CONCESSIONARIA_FIXA = "Equatorial";

export default function LearningForm() {
  const inputRef = useRef<HTMLInputElement>(null);
  const pasteRef = useRef<HTMLDivElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [pasteHint, setPasteHint] = useState(false);

  // Tipo de caso: estrutura (foco nos códigos) ou poste (foco no número + tamanho)
  const [tipoCaso, setTipoCaso] = useState<"estrutura" | "poste">("estrutura");

  // Campos comuns
  const [conformidade, setConformidade] = useState<boolean | null>(null);
  const [notes, setNotes] = useState("");

  // Campos — Estrutura
  const [codesMT, setCodesMT] = useState("");   // Média Tensão
  const [codesBT, setCodesBT] = useState("");   // Baixa Tensão

  // Campos — Poste
  const [poleSize, setPoleSize] = useState("");
  const [poleNumber, setPoleNumber] = useState("");   // número físico visível na foto / placa

  // Anotação visual: bbox confirmada pelo usuário
  const [annotationBbox, setAnnotationBbox] = useState<BBox | null>(null);

  // Label derivado dos campos preenchidos — passado ao AnnotationCanvas
  const activeLabel = useMemo(() => {
    if (tipoCaso === "estrutura") {
      const parts: string[] = [];
      if (codesMT.trim()) parts.push(`MT:${codesMT.trim()}`);
      if (codesBT.trim()) parts.push(`BT:${codesBT.trim()}`);
      return parts.join(" / ");
    } else {
      const sizePart = poleSize.trim();
      const numPart  = poleNumber.trim() ? `Nº${poleNumber.trim()}` : "";
      return [sizePart, numPart].filter(Boolean).join(" ") || "POSTE";
    }
  }, [tipoCaso, codesMT, codesBT, poleSize, poleNumber]);

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const [cases, setCases] = useState<LearningCase[]>([]);
  const [loadingCases, setLoadingCases] = useState(true);

  const loadCases = useCallback(async () => {
    setLoadingCases(true);
    try {
      const r = await fetch("/api/learning");
      if (r.ok) {
        const data = await r.json();
        setCases(data);
      }
    } catch {
      // ignore
    } finally {
      setLoadingCases(false);
    }
  }, []);

  useEffect(() => {
    loadCases();
  }, [loadCases]);

  // Ctrl+V global — captura imagem do clipboard em qualquer lugar da página
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of Array.from(items)) {
        if (item.type.startsWith("image/")) {
          const blob = item.getAsFile();
          if (blob) {
            pickImageBlob(blob);
            // Pisca o hint visual
            setPasteHint(true);
            setTimeout(() => setPasteHint(false), 1200);
          }
          break;
        }
      }
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preview]);

  function pickImageBlob(blob: Blob) {
    if (blob.size > 10 * 1024 * 1024) {
      setError("Imagem supera 10 MB");
      return;
    }
    setError(null);
    const f = new File([blob], "foto_colada.jpg", { type: blob.type || "image/jpeg" });
    setAnnotationBbox(null);
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(blob));
  }

  function pickFile(f: File | null) {
    if (!f) return;
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["jpg", "jpeg", "png"].includes(ext || "")) {
      setError("O arquivo precisa ser JPG ou PNG");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Arquivo supera 10 MB");
      return;
    }
    setError(null);
    setAnnotationBbox(null);
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    pickFile(e.dataTransfer.files?.[0] || null);
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) { setError("Selecione uma foto"); return; }
    if (conformidade === null) { setError("Selecione Conforme ou Não conforme"); return; }
    setError(null);
    setSaving(true);
    setSuccess(false);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("concessionaria", CONCESSIONARIA_FIXA);
    fd.append("conformidade", conformidade ? "true" : "false");
    fd.append("notes", notes);
    if (annotationBbox) {
      fd.append("bbox", JSON.stringify(annotationBbox));
    }

    if (tipoCaso === "estrutura") {
      // Monta lista separando MT e BT com prefixo
      const parts: string[] = [];
      if (codesMT.trim()) parts.push(...codesMT.split(",").map(c => `MT:${c.trim()}`));
      if (codesBT.trim()) parts.push(...codesBT.split(",").map(c => `BT:${c.trim()}`));
      fd.append("structure_codes", parts.length > 0 ? parts.join(", ") : "N/A");
      fd.append("pole_size", "");
    } else {
      // Poste: o "código da estrutura" guarda o número do poste para matching
      fd.append("structure_codes", poleNumber ? `POSTE:${poleNumber}` : "POSTE");
      fd.append("pole_size", poleSize);
    }

    try {
      const r = await fetch("/api/learning", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) {
        setError(data.detail || data.error || "Erro ao salvar");
        return;
      }
      setSuccess(true);
      // Reset form
      setFile(null);
      if (preview) URL.revokeObjectURL(preview);
      setPreview(null);
      setCodesMT("");
      setCodesBT("");
      setPoleSize("");
      setPoleNumber("");
      setConformidade(null);
      setNotes("");
      setAnnotationBbox(null);
      await loadCases();
      setTimeout(() => setSuccess(false), 3000);
    } catch {
      setError("Falha de rede");
    } finally {
      setSaving(false);
    }
  }

  async function deleteCase(id: string) {
    if (!confirm("Remover este caso de treinamento?")) return;
    try {
      await fetch(`/api/learning/${id}`, { method: "DELETE" });
      setCases((prev) => prev.filter((c) => c.id !== id));
    } catch {
      // ignore
    }
  }

  return (
    <>
      <form onSubmit={submit} className="bg-bg-card border border-grid-line rounded-2xl p-6 space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Upload area */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                Foto do campo
              </span>
              {/* Hint Ctrl+V */}
              <span className={`text-xs font-mono px-2 py-0.5 rounded border transition-all duration-300 ${
                pasteHint
                  ? "bg-accent/20 text-accent border-accent/40 scale-105"
                  : "bg-bg-elevated text-slate-500 border-grid-line"
              }`}>
                Ctrl+V para colar
              </span>
            </div>

            {/* input hidden — sempre presente */}
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              className="hidden"
              onChange={(e) => pickFile(e.target.files?.[0] || null)}
            />

            {file ? (
              /* ── Foto carregada: canvas de anotação ── */
              <div className="space-y-2">
                {/* Botão remover foto */}
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => {
                      setFile(null);
                      if (preview) URL.revokeObjectURL(preview);
                      setPreview(null);
                      setAnnotationBbox(null);
                    }}
                    className="text-xs text-slate-500 hover:text-red-400 transition flex items-center gap-1"
                  >
                    × Remover foto
                  </button>
                </div>
                {pasteHint && (
                  <div className="rounded-lg bg-accent/10 border border-accent/30 px-3 py-1.5 text-center text-xs text-accent font-semibold">
                    ✓ Imagem colada!
                  </div>
                )}
                {/* Canvas com detecção por IA */}
                <AnnotationCanvas
                  imageFile={file}
                  label={activeLabel}
                  onAnnotated={setAnnotationBbox}
                />
                {/* Hint: label atual */}
                {activeLabel && (
                  <div className="text-[11px] text-slate-500 text-center">
                    Detectando: <span className="font-mono text-accent">{activeLabel}</span>
                    {" "}— preencha os campos ao lado e clique em Detectar
                  </div>
                )}
              </div>
            ) : (
              /* ── Sem foto: drop zone ── */
              <div
                ref={pasteRef}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                className={`relative cursor-pointer rounded-xl border-2 border-dashed transition flex items-center justify-center overflow-hidden ${
                  pasteHint
                    ? "border-accent bg-accent/10"
                    : dragOver
                    ? "border-accent bg-accent/5"
                    : "border-grid-line bg-bg-elevated hover:border-slate-600"
                }`}
                style={{ minHeight: "240px" }}
              >
                <div className="text-center p-8 select-none">
                  <div className="text-5xl mb-3 opacity-30">📷</div>
                  <div className="font-medium text-slate-200">Arraste ou clique para selecionar</div>
                  <div className="mt-3 inline-flex items-center gap-2 bg-bg-card border border-grid-line rounded-lg px-4 py-2">
                    <kbd className="text-xs font-mono text-accent font-bold">Ctrl</kbd>
                    <span className="text-slate-600 text-xs">+</span>
                    <kbd className="text-xs font-mono text-accent font-bold">V</kbd>
                    <span className="text-xs text-slate-400 ml-1">para colar direto do clipboard</span>
                  </div>
                  <div className="text-xs text-slate-600 mt-3">JPG / PNG · máx. 10 MB</div>
                </div>
              </div>
            )}
          </div>

          {/* Form fields */}
          <div className="space-y-4">

            {/* Seletor de tipo de caso */}
            <div>
              <span className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-2 block">
                O que esta foto mostra?
              </span>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setTipoCaso("estrutura")}
                  className={`py-2.5 px-3 rounded-xl border-2 font-semibold text-sm transition text-left ${
                    tipoCaso === "estrutura"
                      ? "border-accent bg-accent/10 text-accent"
                      : "border-grid-line bg-bg-elevated text-slate-400 hover:border-slate-500"
                  }`}
                >
                  <div>⚡ Estrutura</div>
                  <div className="text-[10px] font-normal opacity-70 mt-0.5">N1, UP1+U3, S3I…</div>
                </button>
                <button
                  type="button"
                  onClick={() => setTipoCaso("poste")}
                  className={`py-2.5 px-3 rounded-xl border-2 font-semibold text-sm transition text-left ${
                    tipoCaso === "poste"
                      ? "border-amber-500 bg-amber-500/10 text-amber-400"
                      : "border-grid-line bg-bg-elevated text-slate-400 hover:border-slate-500"
                  }`}
                >
                  <div>🪵 Poste</div>
                  <div className="text-[10px] font-normal opacity-70 mt-0.5">Número + tamanho</div>
                </button>
              </div>
            </div>

            {/* Campos condicionais por tipo */}
            {tipoCaso === "estrutura" ? (
              <div className="space-y-3">
                {/* MT */}
                <div>
                  <label className="block">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="w-2 h-2 rounded-full bg-orange-400 flex-shrink-0" />
                      <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                        Estrutura MT
                      </span>
                      <span className="text-[10px] text-slate-600">(Média Tensão)</span>
                    </div>
                    <input
                      value={codesMT}
                      onChange={(e) => setCodesMT(e.target.value)}
                      className="form-input"
                      placeholder="ex: N1, UP1+U3, S3I  — deixe vazio se não tiver"
                      autoFocus
                    />
                  </label>
                </div>

                {/* BT */}
                <div>
                  <label className="block">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="w-2 h-2 rounded-full bg-blue-400 flex-shrink-0" />
                      <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                        Estrutura BT
                      </span>
                      <span className="text-[10px] text-slate-600">(Baixa Tensão)</span>
                    </div>
                    <input
                      value={codesBT}
                      onChange={(e) => setCodesBT(e.target.value)}
                      className="form-input"
                      placeholder="ex: R1, R3, BT-01  — deixe vazio se não tiver"
                    />
                  </label>
                </div>

                {/* Indicador do que foi preenchido */}
                {(codesMT || codesBT) && (
                  <div className="flex gap-2 flex-wrap">
                    {codesMT && (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-orange-500/15 border border-orange-500/30 text-orange-300 font-mono">
                        MT: {codesMT}
                      </span>
                    )}
                    {codesBT && (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/15 border border-blue-500/30 text-blue-300 font-mono">
                        BT: {codesBT}
                      </span>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <>
                <Field label="Tamanho do poste">
                  <input
                    value={poleSize}
                    onChange={(e) => setPoleSize(e.target.value)}
                    className="form-input"
                    placeholder="ex: 11/300, 9/150, 13/600"
                    autoFocus
                  />
                </Field>
                <Field label="Número do poste (visível na foto / placa)">
                  <input
                    value={poleNumber}
                    onChange={(e) => setPoleNumber(e.target.value)}
                    className="form-input"
                    placeholder="ex: 001, P-23, 4872"
                  />
                </Field>
                <div className="rounded-lg bg-amber-950/30 border border-amber-800/40 px-3 py-2 text-xs text-amber-300">
                  💡 O agente vai verificar se este número bate com o nome do placemark no KMZ.
                </div>
              </>
            )}

            {/* Concessionária fixa */}
            <div>
              <span className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-1.5 block">
                Concessionária
              </span>
              <div className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-grid-line bg-bg-elevated">
                <span className="w-2 h-2 rounded-full bg-accent flex-shrink-0" />
                <span className="text-sm text-white font-medium">{CONCESSIONARIA_FIXA}</span>
              </div>
            </div>

            <div>
              <span className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-2 block">
                Conformidade
              </span>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setConformidade(true)}
                  className={`py-3 rounded-xl border-2 font-semibold text-sm transition ${
                    conformidade === true
                      ? "border-emerald-500 bg-emerald-500/15 text-emerald-300"
                      : "border-grid-line bg-bg-elevated text-slate-400 hover:border-emerald-700 hover:text-emerald-400"
                  }`}
                >
                  ✅ Conforme
                </button>
                <button
                  type="button"
                  onClick={() => setConformidade(false)}
                  className={`py-3 rounded-xl border-2 font-semibold text-sm transition ${
                    conformidade === false
                      ? "border-red-500 bg-red-500/15 text-red-300"
                      : "border-grid-line bg-bg-elevated text-slate-400 hover:border-red-700 hover:text-red-400"
                  }`}
                >
                  ❌ Não conforme
                </button>
              </div>
            </div>

            <Field label="Observações">
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
                className="form-input resize-none"
                placeholder="Descreva o que está correto ou incorreto nesta foto"
              />
            </Field>
          </div>
        </div>

        {error && (
          <div className="rounded-lg bg-red-950/40 border border-red-900/60 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}
        {success && (
          <div className="rounded-lg bg-emerald-950/40 border border-emerald-900/60 px-3 py-2 text-sm text-emerald-300">
            Caso de treinamento salvo com sucesso.
          </div>
        )}

        <button
          type="submit"
          disabled={saving}
          className="w-full bg-accent hover:bg-accent-hover disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-bg font-bold py-3 rounded-xl transition"
        >
          {saving ? "Salvando…" : "Salvar caso de treinamento"}
        </button>
      </form>

      {/* Gallery */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">
            Casos salvos{" "}
            {cases.length > 0 && (
              <span className="text-sm text-slate-500 font-normal">({cases.length})</span>
            )}
          </h3>
          <button
            onClick={loadCases}
            className="text-xs text-accent hover:underline"
          >
            Atualizar
          </button>
        </div>

        {loadingCases ? (
          <div className="text-center py-10 text-slate-500">Carregando…</div>
        ) : cases.length === 0 ? (
          <div className="bg-bg-card border border-grid-line border-dashed rounded-xl p-12 text-center">
            <div className="text-4xl mb-3 opacity-30">🎓</div>
            <p className="text-slate-400 text-sm">Nenhum caso de treinamento ainda.</p>
            <p className="text-slate-600 text-xs mt-1">Envie a primeira foto acima.</p>
          </div>
        ) : (
          <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {cases.map((c) => (
              <CaseCard key={c.id} c={c} onDelete={deleteCase} />
            ))}
          </div>
        )}
      </div>

      <style jsx global>{`
        .form-input {
          width: 100%;
          background-color: #0a0e1a;
          border: 1px solid #1e293b;
          color: #e5e7eb;
          padding: 0.625rem 0.875rem;
          border-radius: 0.5rem;
          outline: none;
          transition: border-color 0.15s, box-shadow 0.15s;
        }
        .form-input:focus {
          border-color: #22d3ee;
          box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15);
        }
      `}</style>
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-1.5 block">
        {label}
      </span>
      {children}
    </label>
  );
}

function CaseCard({
  c,
  onDelete,
}: {
  c: LearningCase;
  onDelete: (id: string) => void;
}) {
  const isPoste = c.structure_codes.some(s => s.startsWith("POSTE"));
  const mtCodes = c.structure_codes.filter(s => s.startsWith("MT:")).map(s => s.replace("MT:", ""));
  const btCodes = c.structure_codes.filter(s => s.startsWith("BT:")).map(s => s.replace("BT:", ""));
  const rawCodes = c.structure_codes.filter(s => !s.startsWith("MT:") && !s.startsWith("BT:") && !s.startsWith("POSTE"));
  const codesPoste = isPoste
    ? c.structure_codes.join(", ").replace("POSTE:", "Nº ") + (c.pole_size ? ` · ${c.pole_size}` : "")
    : "";
  const date = new Date(c.created_at).toLocaleDateString("pt-BR");

  return (
    <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden group relative">
      {/* Thumbnail */}
      <div className="relative h-36 bg-bg-elevated overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`/api/learning/${c.id}/image`}
          alt={isPoste ? codesPoste : [...mtCodes, ...btCodes, ...rawCodes].join(", ") || "foto"}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
        {/* Delete button */}
        <button
          onClick={() => onDelete(c.id)}
          className="absolute top-1.5 right-1.5 bg-black/60 hover:bg-red-900/90 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition"
          title="Remover caso"
        >
          ×
        </button>
        {/* Conformidade badge */}
        <span
          className={`absolute bottom-1.5 left-1.5 text-[10px] font-bold px-1.5 py-0.5 rounded-md border ${
            c.conformidade
              ? "bg-emerald-900/80 text-emerald-300 border-emerald-700/50"
              : "bg-red-900/80 text-red-300 border-red-700/50"
          }`}
        >
          {c.conformidade ? "Conforme" : "Não conforme"}
        </span>
      </div>

      {/* Info */}
      <div className="p-3 space-y-1.5">
        <span className="text-[10px] px-1.5 py-0.5 rounded border font-semibold bg-bg-elevated border-grid-line text-slate-500">
          {isPoste ? "🪵 Poste" : "⚡ Estrutura"}
        </span>

        {isPoste && (
          <div className="text-xs font-mono text-amber-300 truncate">{codesPoste}</div>
        )}
        {mtCodes.length > 0 && (
          <div className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-orange-400 flex-shrink-0" />
            <span className="text-[11px] font-mono text-orange-300 truncate">MT: {mtCodes.join(", ")}</span>
          </div>
        )}
        {btCodes.length > 0 && (
          <div className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 flex-shrink-0" />
            <span className="text-[11px] font-mono text-blue-300 truncate">BT: {btCodes.join(", ")}</span>
          </div>
        )}
        {rawCodes.length > 0 && (
          <div className="text-[11px] font-mono text-accent truncate">{rawCodes.join(", ")}</div>
        )}
        <div className="text-[10px] text-slate-600">{date}</div>
      </div>
    </div>
  );
}
