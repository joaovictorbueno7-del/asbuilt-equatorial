"use client";

import { useState, useRef, DragEvent } from "react";
import { useRouter } from "next/navigation";

const CONCESSIONARIAS = ["Equatorial", "Cemig", "Copel", "Enel", "Light", "EDP", "Energisa", "Outro"];
const PARALLELISM = 3;

type Status = "queued" | "uploading" | "extracting" | "done" | "duplicate" | "failed";

type Item = {
  file: File;
  codigo: string;
  nome: string;
  versao: string;
  status: Status;
  error?: string;
  norm_id?: string;
  structures?: number;
};

const STATUS_BADGE: Record<Status, string> = {
  queued: "bg-slate-500/20 text-slate-300 border-slate-500/40",
  uploading: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 animate-pulse",
  extracting: "bg-amber-500/20 text-amber-300 border-amber-500/40 animate-pulse",
  done: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  duplicate: "bg-purple-500/20 text-purple-300 border-purple-500/40",
  failed: "bg-red-500/20 text-red-300 border-red-500/40",
};

const STATUS_LABEL: Record<Status, string> = {
  queued: "Aguardando", uploading: "Enviando", extracting: "Extraindo estruturas",
  done: "Concluída", duplicate: "Duplicata (já existe)", failed: "Falhou",
};

function autoCodigoFromFilename(name: string): string {
  const stem = name.replace(/\.pdf$/i, "");
  const m = stem.match(/^([A-Za-z]{1,5}[._-]\d+[._-][A-Za-z]+(?:[._-]\d+)?)/);
  if (m) return m[1].toUpperCase().replace(/_/g, ".");
  return stem.slice(0, 40).toUpperCase();
}

function autoNomeFromFilename(name: string): string {
  const stem = name.replace(/\.pdf$/i, "");
  return stem.replace(/[._-]/g, " ").replace(/\s+/g, " ").trim();
}

export default function UploadForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [concessionaria, setConcessionaria] = useState(CONCESSIONARIAS[0]);
  const [versao, setVersao] = useState("1.0");
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);

  function addFiles(fileList: FileList | null) {
    if (!fileList) return;
    const incoming: Item[] = [];
    for (const f of Array.from(fileList)) {
      if (!f.name.toLowerCase().endsWith(".pdf")) continue;
      if (f.size > 100 * 1024 * 1024) continue;
      incoming.push({
        file: f,
        codigo: autoCodigoFromFilename(f.name),
        nome: autoNomeFromFilename(f.name),
        versao,
        status: "queued",
      });
    }
    setItems((prev) => [...prev, ...incoming]);
    setDone(false);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  }

  function updateItem(idx: number, patch: Partial<Item>) {
    setItems((prev) => prev.map((it, i) => i === idx ? { ...it, ...patch } : it));
  }

  function removeItem(idx: number) {
    setItems((prev) => prev.filter((_, i) => i !== idx));
  }

  async function processOne(idx: number, item: Item) {
    updateItem(idx, { status: "uploading" });
    const fd = new FormData();
    fd.append("file", item.file);
    fd.append("concessionaria", concessionaria);
    fd.append("codigo", item.codigo);
    fd.append("nome", item.nome);
    fd.append("versao", item.versao || versao);
    try {
      const r = await fetch("/api/knowledge", { method: "POST", body: fd });
      const data = await r.json();
      if (r.status === 409) {
        updateItem(idx, { status: "duplicate", error: data.detail || "PDF já cadastrado" });
        return;
      }
      if (!r.ok) {
        updateItem(idx, { status: "failed", error: data.detail || data.error || `HTTP ${r.status}` });
        return;
      }
      updateItem(idx, { status: "extracting", norm_id: data.id });
      for (let i = 0; i < 30; i++) {
        await new Promise((res) => setTimeout(res, 3000));
        const dr = await fetch(`/api/knowledge/${data.id}`, { cache: "no-store" });
        if (dr.ok) {
          const dd = await dr.json();
          if (dd.structure_count > 0) {
            updateItem(idx, { status: "done", structures: dd.structure_count });
            return;
          }
        }
      }
      updateItem(idx, { status: "done", structures: 0 });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "erro desconhecido";
      updateItem(idx, { status: "failed", error: msg });
    }
  }

  async function startBatch() {
    if (running || items.length === 0) return;
    setRunning(true);
    setDone(false);
    const queue = items.map((_, idx) => idx);
    let cursor = 0;
    async function worker() {
      while (cursor < queue.length) {
        const my = cursor++;
        const idx = queue[my];
        const latest = items[idx];
        if (!latest) continue;
        await processOne(idx, latest);
      }
    }
    await Promise.all(Array.from({ length: PARALLELISM }, () => worker()));
    setRunning(false);
    setDone(true);
    router.refresh();
  }

  const summary = {
    total: items.length,
    done: items.filter((i) => i.status === "done").length,
    duplicates: items.filter((i) => i.status === "duplicate").length,
    failed: items.filter((i) => i.status === "failed").length,
    structures: items.reduce((s, i) => s + (i.structures || 0), 0),
  };

  return (
    <div className="space-y-5">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !running && inputRef.current?.click()}
        className={`cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition ${
          dragOver ? "border-accent bg-accent/5" : "border-grid-line bg-bg-card hover:border-slate-600"
        } ${running ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <input ref={inputRef} type="file" accept=".pdf" multiple className="hidden"
               onChange={(e) => addFiles(e.target.files)} />
        <div className="text-4xl mb-2 opacity-40">⬆️</div>
        <div className="font-semibold text-slate-200">Arraste vários PDFs ou clique para escolher</div>
        <div className="text-xs text-slate-600 mt-1">processamento paralelo · até {PARALLELISM} ao mesmo tempo</div>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <Field label="Concessionária (aplicada a todos)">
          <select value={concessionaria} onChange={(e) => setConcessionaria(e.target.value)}
                  disabled={running} className="form-input">
            {CONCESSIONARIAS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Versão padrão">
          <input value={versao} onChange={(e) => setVersao(e.target.value)} disabled={running} className="form-input" />
        </Field>
      </div>

      {items.length > 0 && (
        <div className="bg-bg-card border border-grid-line rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-grid-line flex items-center justify-between bg-bg-elevated/40">
            <div className="text-sm font-semibold">Fila ({items.length} PDFs)</div>
            <div className="text-xs text-slate-500">
              {summary.done > 0 && <span className="text-emerald-400 mr-3">✓ {summary.done}</span>}
              {summary.duplicates > 0 && <span className="text-purple-400 mr-3">⊘ {summary.duplicates} dup</span>}
              {summary.failed > 0 && <span className="text-red-400 mr-3">✗ {summary.failed}</span>}
            </div>
          </div>
          <div className="divide-y divide-grid-line/40 max-h-[480px] overflow-auto">
            {items.map((it, idx) => (
              <div key={idx} className="px-4 py-3 grid grid-cols-12 gap-3 items-center text-sm">
                <div className="col-span-5">
                  <div className="font-medium text-white truncate" title={it.file.name}>{it.file.name}</div>
                  <div className="text-[10px] text-slate-500">{(it.file.size / 1024).toFixed(0)} KB</div>
                </div>
                <input
                  className="form-input col-span-2 text-xs"
                  value={it.codigo}
                  disabled={running || it.status !== "queued"}
                  onChange={(e) => updateItem(idx, { codigo: e.target.value })}
                  placeholder="código"
                />
                <input
                  className="form-input col-span-3 text-xs"
                  value={it.nome}
                  disabled={running || it.status !== "queued"}
                  onChange={(e) => updateItem(idx, { nome: e.target.value })}
                  placeholder="nome"
                />
                <div className="col-span-2 text-right">
                  <span className={`text-[10px] uppercase font-semibold px-2 py-0.5 rounded border ${STATUS_BADGE[it.status]}`}>
                    {STATUS_LABEL[it.status]}
                  </span>
                  {it.status === "done" && it.structures !== undefined && (
                    <div className="text-[10px] text-emerald-400 mt-0.5">{it.structures} estruturas</div>
                  )}
                  {it.status === "failed" && it.error && (
                    <div className="text-[10px] text-red-400 mt-0.5 truncate" title={it.error}>{it.error.slice(0, 50)}</div>
                  )}
                  {it.status === "duplicate" && (
                    <div className="text-[10px] text-purple-400 mt-0.5">já cadastrado</div>
                  )}
                  {it.status === "queued" && !running && (
                    <button onClick={() => removeItem(idx)} className="text-[10px] text-slate-600 hover:text-red-400 mt-0.5">remover</button>
                  )}
                </div>
                <div className="col-span-12">
                  {it.status === "uploading" && <ProgressBar pct={50} color="bg-cyan-500" />}
                  {it.status === "extracting" && <ProgressBar pct={80} color="bg-amber-500" />}
                  {it.status === "done" && <ProgressBar pct={100} color="bg-emerald-500" />}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-3">
        <button onClick={startBatch} disabled={running || items.length === 0}
                className="flex-1 bg-accent hover:bg-accent-hover disabled:bg-slate-700 disabled:text-slate-500 text-bg font-bold py-3 rounded-xl transition">
          {running ? "Processando..." : `Processar ${items.length} PDF${items.length !== 1 ? "s" : ""} em paralelo`}
        </button>
        {!running && items.length > 0 && (
          <button onClick={() => { setItems([]); setDone(false); }}
                  className="px-4 py-3 rounded-xl border border-grid-line text-slate-300 hover:bg-bg-elevated">
            Limpar
          </button>
        )}
      </div>

      {done && (
        <div className="bg-emerald-950/30 border border-emerald-900/50 rounded-xl p-4 text-sm">
          <div className="font-semibold text-emerald-200 mb-1">✅ Lote concluído</div>
          <div className="text-emerald-300/80">
            <strong>{summary.done}</strong> norma(s) adicionada(s) · <strong>{summary.structures}</strong> estruturas indexadas
            {summary.duplicates > 0 && <> · <span className="text-purple-300">{summary.duplicates} duplicata(s)</span></>}
            {summary.failed > 0 && <> · <span className="text-red-300">{summary.failed} falha(s)</span></>}
          </div>
          <a href="/knowledge" className="inline-block mt-2 text-accent hover:underline text-xs">Ver lista completa →</a>
        </div>
      )}

      <p className="text-xs text-slate-500 text-center">
        🔒 PDFs arquivados em <code className="text-slate-400">backend/knowledge/normas/</code> · nunca deletados · backup diário
      </p>

      <style jsx global>{`
        .form-input { width: 100%; background-color: #0a0e1a; border: 1px solid #1e293b; color: #e5e7eb;
                      padding: 0.4rem 0.6rem; border-radius: 0.4rem; outline: none; font-size: 0.875rem; }
        .form-input:focus { border-color: #22d3ee; box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15); }
        .form-input:disabled { opacity: 0.5; cursor: not-allowed; }
      `}</style>
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

function ProgressBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="h-1 bg-bg-elevated rounded-full overflow-hidden mt-2">
      <div className={`h-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
    </div>
  );
}
