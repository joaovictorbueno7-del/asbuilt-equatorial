"use client";

import { useState, useRef, useEffect, DragEvent } from "react";
import { useRouter } from "next/navigation";

// ── Tipos ────────────────────────────────────────────────────────────────────

type RecognizeResult = {
  tipo: string;
  codigo: string | null;
  tamanho: string | null;
  conformidade: boolean | null;
  confianca: number;
  descricao: string;
  observacoes: string;
  n_cases_used: number;
  visual_examples?: number;
};

type Tab = "foto" | "kmz";

// ── Helpers ──────────────────────────────────────────────────────────────────

const TIPO_INFO: Record<string, { icon: string; label: string; color: string }> = {
  poste:           { icon: "🪵", label: "Poste",           color: "text-amber-300" },
  estrutura_mt:    { icon: "⚡", label: "Estrutura MT",    color: "text-orange-300" },
  estrutura_bt:    { icon: "⚡", label: "Estrutura BT",    color: "text-blue-300" },
  estrutura_mt_bt: { icon: "⚡", label: "Estrutura MT+BT", color: "text-purple-300" },
  desconhecido:    { icon: "❓", label: "Não identificado", color: "text-slate-400" },
};

// ── Componente principal ──────────────────────────────────────────────────────

export default function TestForm() {
  const [tab, setTab] = useState<Tab>("foto");
  return (
    <div className="space-y-0">
      {/* Tab bar */}
      <div className="flex border-b border-grid-line mb-6">
        <TabBtn active={tab === "foto"} onClick={() => setTab("foto")} icon="📸">
          Testar com Foto
        </TabBtn>
        <TabBtn active={tab === "kmz"} onClick={() => setTab("kmz")} icon="🗺️">
          Importar KMZ de Teste
        </TabBtn>
      </div>

      {tab === "foto" ? <FotoTab /> : <KmzTab />}
    </div>
  );
}

function TabBtn({ active, onClick, icon, children }: {
  active: boolean; onClick: () => void; icon: string; children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-5 py-2.5 text-sm font-semibold border-b-2 transition -mb-px ${
        active
          ? "border-accent text-accent"
          : "border-transparent text-slate-500 hover:text-slate-300"
      }`}
    >
      <span>{icon}</span>
      {children}
    </button>
  );
}

// ── Tab 1: Reconhecimento por foto ────────────────────────────────────────────

function FotoTab() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile]       = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [pasteHint, setPasteHint] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult]       = useState<RecognizeResult | null>(null);
  const [error, setError]         = useState<string | null>(null);

  // Ctrl+V global
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      if ((e.target as HTMLElement)?.tagName === "INPUT" ||
          (e.target as HTMLElement)?.tagName === "TEXTAREA") return;
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of Array.from(items)) {
        if (item.type.startsWith("image/")) {
          const blob = item.getAsFile();
          if (blob) { pickBlob(blob); setPasteHint(true); setTimeout(() => setPasteHint(false), 1500); }
          break;
        }
      }
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function pickBlob(blob: Blob) {
    if (blob.size > 10 * 1024 * 1024) { setError("Imagem supera 10 MB"); return; }
    setError(null); setResult(null);
    const f = new File([blob], "foto_teste.jpg", { type: blob.type || "image/jpeg" });
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(blob));
  }

  function pickFile(f: File | null) {
    if (!f) return;
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["jpg","jpeg","png"].includes(ext || "")) { setError("Use JPG ou PNG"); return; }
    if (f.size > 10 * 1024 * 1024) { setError("Arquivo supera 10 MB"); return; }
    setError(null); setResult(null);
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
  }

  function onDrop(e: DragEvent) {
    e.preventDefault(); setDragOver(false);
    pickFile(e.dataTransfer.files?.[0] || null);
  }

  function clear() {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null); setResult(null); setError(null);
  }

  // Análise automática quando file muda
  useEffect(() => {
    if (file) runAnalyze(file);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  async function runAnalyze(f: File) {
    setAnalyzing(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch("/api/learning/recognize", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) {
        setError(data.detail || data.error || `Erro ${r.status} — tente reiniciar o backend`);
        return;
      }
      setResult(data as RecognizeResult);
    } catch {
      setError("Falha de rede — verifique se o backend está rodando");
    } finally {
      setAnalyzing(false);
    }
  }

  const tipoInfo = result ? (TIPO_INFO[result.tipo] ?? TIPO_INFO.desconhecido) : null;
  const pct = result ? Math.round(result.confianca * 100) : 0;
  const confColor    = pct >= 80 ? "text-emerald-400"  : pct >= 55 ? "text-amber-400"  : "text-red-400";
  const confBarColor = pct >= 80 ? "bg-emerald-500"    : pct >= 55 ? "bg-amber-500"    : "bg-red-500";

  return (
    <div className="space-y-6">
      {/* Drop zone / Preview */}
      <div className="bg-bg-card border border-grid-line rounded-2xl p-6">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">Foto para testar</span>
          <span className={`text-xs font-mono px-2 py-0.5 rounded border transition-all duration-300 ${
            pasteHint ? "bg-accent/20 text-accent border-accent/40 scale-105" : "bg-bg-elevated text-slate-500 border-grid-line"
          }`}>Ctrl+V para colar</span>
        </div>

        <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" className="hidden"
          onChange={(e) => pickFile(e.target.files?.[0] || null)} />

        {preview ? (
          <div className="relative rounded-xl overflow-hidden bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="foto" className="w-full max-h-72 object-contain" />
            <button onClick={clear}
              className="absolute top-2 right-2 bg-black/60 hover:bg-red-900/80 text-white rounded-full w-8 h-8 flex items-center justify-center text-sm transition">
              ×
            </button>
            {pasteHint && (
              <div className="absolute inset-0 flex items-center justify-center bg-accent/10">
                <span className="text-accent font-bold text-sm bg-black/70 px-4 py-2 rounded-xl">✓ Imagem colada!</span>
              </div>
            )}
          </div>
        ) : (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`cursor-pointer rounded-xl border-2 border-dashed transition flex flex-col items-center justify-center gap-4 py-14 ${
              pasteHint || dragOver ? "border-accent bg-accent/5" : "border-grid-line bg-bg-elevated hover:border-slate-500"
            }`}
          >
            <div className="text-5xl opacity-20">📸</div>
            <div className="text-center">
              <div className="font-semibold text-slate-200 text-sm mb-2">Cole, arraste ou clique</div>
              <div className="inline-flex items-center gap-2 bg-bg-card border border-grid-line rounded-lg px-4 py-2">
                <kbd className="text-xs font-mono text-accent font-bold">Ctrl</kbd>
                <span className="text-slate-600 text-xs">+</span>
                <kbd className="text-xs font-mono text-accent font-bold">V</kbd>
                <span className="text-xs text-slate-400 ml-1">para colar do clipboard</span>
              </div>
              <div className="text-xs text-slate-600 mt-2">Análise começa automaticamente</div>
            </div>
          </div>
        )}
      </div>

      {/* Analisando */}
      {analyzing && (
        <div className="bg-bg-card border border-grid-line rounded-2xl p-8 flex flex-col items-center gap-4">
          <div className="relative w-14 h-14">
            <div className="absolute inset-0 rounded-full border-4 border-accent/20" />
            <div className="absolute inset-0 rounded-full border-4 border-t-accent animate-spin" />
            <div className="absolute inset-0 flex items-center justify-center text-xl">🔍</div>
          </div>
          <div className="text-center">
            <div className="text-white font-semibold">Analisando com IA…</div>
            <div className="text-slate-400 text-xs mt-1">Comparando com os casos de treinamento</div>
          </div>
        </div>
      )}

      {/* Resultado */}
      {result && !analyzing && (
        <div className="bg-bg-card border border-grid-line rounded-2xl overflow-hidden">
          {/* Header */}
          <div className="px-6 pt-5 pb-4 border-b border-grid-line flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-3xl">{tipoInfo?.icon}</span>
              <div>
                <div className={`text-lg font-bold ${tipoInfo?.color}`}>{tipoInfo?.label}</div>
                {result.codigo && (
                  <div className="text-white font-mono text-sm mt-0.5">
                    {result.tipo === "poste" ? "Nº " : ""}{result.codigo}
                    {result.tamanho && <span className="text-slate-400 ml-2">· {result.tamanho}</span>}
                  </div>
                )}
              </div>
            </div>
            <div className="flex-shrink-0">
              {result.conformidade === true  && <Badge color="emerald">✅ Conforme</Badge>}
              {result.conformidade === false && <Badge color="red">❌ Não conforme</Badge>}
              {result.conformidade === null  && <Badge color="slate">— Sem avaliação</Badge>}
            </div>
          </div>

          {/* Body */}
          <div className="px-6 py-5 space-y-4">
            {/* Confiança */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs uppercase tracking-wider text-slate-500 font-medium">Confiança</span>
                <span className={`text-sm font-bold font-mono ${confColor}`}>{pct}%</span>
              </div>
              <div className="h-2 rounded-full bg-bg-elevated overflow-hidden">
                <div className={`h-full rounded-full transition-all duration-700 ${confBarColor}`}
                  style={{ width: `${pct}%` }} />
              </div>
              <div className="text-[11px] text-slate-600 mt-1">
                {pct >= 80 ? "Alta confiança — resultado confiável"
                  : pct >= 55 ? "Confiança média — revise se necessário"
                  : "Baixa confiança — foto ruim ou elemento não reconhecido"}
              </div>
            </div>

            {/* Descrição */}
            {result.descricao && (
              <InfoBox label="O que foi visto">{result.descricao}</InfoBox>
            )}

            {/* Observações */}
            {result.observacoes && (
              <InfoBox label="Observações" muted>{result.observacoes}</InfoBox>
            )}

            {/* Rodapé */}
            <div className="flex items-center justify-between pt-1 border-t border-grid-line">
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-600">
                  🎓 {result.n_cases_used} caso{result.n_cases_used !== 1 ? "s" : ""} de treinamento
                  {result.visual_examples !== undefined && result.visual_examples > 0 && (
                    <span className="ml-1 text-accent">
                      ({result.visual_examples} foto{result.visual_examples !== 1 ? "s" : ""} usadas como exemplo)
                    </span>
                  )}
                </span>
                {result.n_cases_used === 0 && (
                  <a href="/treinar" className="text-xs text-accent hover:underline">→ Adicionar casos</a>
                )}
              </div>
              <button onClick={() => file && runAnalyze(file)}
                className="text-xs text-accent hover:underline">
                Analisar novamente
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Erro */}
      {error && (
        <div className="rounded-xl bg-red-950/40 border border-red-900/60 px-4 py-3 space-y-1">
          <div className="text-sm font-semibold text-red-300">Erro na análise</div>
          <div className="text-xs text-red-400 font-mono">{error}</div>
        </div>
      )}

      {/* Estado vazio */}
      {!file && !analyzing && !result && !error && (
        <div className="rounded-xl bg-bg-card border border-grid-line border-dashed p-6 text-center space-y-2">
          <div className="text-3xl opacity-20">🤖</div>
          <div className="text-sm text-slate-400">Cole uma foto e o agente dirá o que reconheceu</div>
          <div className="text-xs text-slate-600">
            Quanto mais casos em{" "}
            <a href="/treinar" className="text-accent hover:underline">Treinar</a>
            , mais preciso fica o reconhecimento
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab 2: Import KMZ de teste ────────────────────────────────────────────────

function KmzTab() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [kmzFile, setKmzFile]   = useState<File | null>(null);
  const [workName, setWorkName] = useState("");
  const [municipio, setMunicipio] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError]       = useState<string | null>(null);

  function pickKmz(f: File | null) {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".kmz")) { setError("O arquivo precisa ser .kmz"); return; }
    setError(null);
    setKmzFile(f);
    if (!workName) setWorkName(f.name.replace(/\.kmz$/i, ""));
  }

  function onDrop(e: DragEvent) {
    e.preventDefault(); setDragOver(false);
    pickKmz(e.dataTransfer.files?.[0] || null);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!kmzFile) { setError("Selecione um arquivo KMZ"); return; }
    if (!workName.trim()) { setError("Informe um nome para o trabalho"); return; }
    setSubmitting(true); setError(null);

    const fd = new FormData();
    fd.append("file", kmzFile);
    fd.append("work_name", workName.trim() + " [TESTE]");
    fd.append("concessionaria", "Equatorial");
    fd.append("tipo", "as_built");
    fd.append("municipio", municipio.trim());

    try {
      const r = await fetch("/api/works", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) {
        setError(data.detail || data.error || `Erro ${r.status}`);
        return;
      }
      // Redireciona para a página da pipeline
      router.push(`/pipelines/${data.run_id}`);
    } catch {
      setError("Falha de rede");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-6">
      <div className="bg-bg-card border border-grid-line rounded-2xl p-6 space-y-5">
        <div>
          <p className="text-sm text-slate-400 mb-4">
            Importe um KMZ e o sistema processará as fotos automaticamente, usando seus casos
            de treinamento para reconhecer estruturas e postes.
          </p>

          {/* Drop zone KMZ */}
          <input ref={inputRef} type="file" accept=".kmz" className="hidden"
            onChange={(e) => pickKmz(e.target.files?.[0] || null)} />

          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !kmzFile && inputRef.current?.click()}
            className={`rounded-xl border-2 border-dashed transition cursor-pointer ${
              kmzFile
                ? "border-emerald-600/50 bg-emerald-950/20 cursor-default"
                : dragOver
                ? "border-accent bg-accent/5"
                : "border-grid-line bg-bg-elevated hover:border-slate-500"
            } flex items-center justify-center gap-4 py-10`}
          >
            {kmzFile ? (
              <div className="text-center">
                <div className="text-2xl mb-1">🗺️</div>
                <div className="text-emerald-300 font-semibold text-sm">{kmzFile.name}</div>
                <div className="text-slate-500 text-xs mt-0.5">
                  {(kmzFile.size / 1024).toFixed(0)} KB
                </div>
                <button
                  type="button"
                  onClick={(ev) => { ev.stopPropagation(); setKmzFile(null); setWorkName(""); }}
                  className="text-xs text-slate-600 hover:text-red-400 mt-2 transition"
                >
                  × Remover
                </button>
              </div>
            ) : (
              <div className="text-center">
                <div className="text-4xl mb-2 opacity-20">🗺️</div>
                <div className="text-sm font-semibold text-slate-300">Arraste o KMZ aqui ou clique</div>
                <div className="text-xs text-slate-600 mt-1">.kmz · máx. 100 MB</div>
              </div>
            )}
          </div>
        </div>

        {/* Nome do trabalho */}
        <div>
          <label className="block text-xs uppercase tracking-wider text-slate-400 font-medium mb-1.5">
            Nome do trabalho
          </label>
          <input
            value={workName}
            onChange={(e) => setWorkName(e.target.value)}
            placeholder="ex: Rota 01 - Hidrolândia"
            className="w-full bg-bg-elevated border border-grid-line text-slate-200 px-3 py-2.5 rounded-lg outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 transition text-sm"
          />
          <div className="text-[11px] text-slate-600 mt-1">
            O sufixo [TESTE] será adicionado automaticamente
          </div>
        </div>

        {/* Município */}
        <div>
          <label className="block text-xs uppercase tracking-wider text-slate-400 font-medium mb-1.5">
            Município <span className="text-slate-600 font-normal">(opcional)</span>
          </label>
          <input
            value={municipio}
            onChange={(e) => setMunicipio(e.target.value)}
            placeholder="ex: Hidrolândia - GO"
            className="w-full bg-bg-elevated border border-grid-line text-slate-200 px-3 py-2.5 rounded-lg outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 transition text-sm"
          />
        </div>

        {/* Informativo */}
        <div className="rounded-lg bg-accent/5 border border-accent/20 px-4 py-3 text-xs text-slate-400 space-y-1">
          <div className="font-semibold text-accent text-[11px] uppercase tracking-wider mb-1">O que acontece ao importar</div>
          <div>1. O KMZ é processado — placemarks e fotos são extraídos</div>
          <div>2. Cada foto é analisada com IA usando seus casos de treinamento</div>
          <div>3. Estruturas e postes são identificados e comparados com as normas</div>
          <div>4. Você acompanha o progresso em tempo real na página do pipeline</div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl bg-red-950/40 border border-red-900/60 px-4 py-3 space-y-1">
          <div className="text-sm font-semibold text-red-300">Erro</div>
          <div className="text-xs text-red-400 font-mono">{error}</div>
        </div>
      )}

      <button
        type="submit"
        disabled={submitting || !kmzFile}
        className="w-full bg-accent hover:bg-accent-hover disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-bg font-bold py-3 rounded-xl transition flex items-center justify-center gap-2"
      >
        {submitting ? (
          <>
            <span className="w-4 h-4 border-2 border-bg/40 border-t-bg rounded-full animate-spin" />
            Enviando KMZ…
          </>
        ) : (
          "🚀 Importar e processar KMZ"
        )}
      </button>
    </form>
  );
}

// ── Componentes auxiliares ────────────────────────────────────────────────────

function Badge({ color, children }: { color: "emerald" | "red" | "slate"; children: React.ReactNode }) {
  const cls = {
    emerald: "bg-emerald-950/50 border-emerald-700/40 text-emerald-300",
    red:     "bg-red-950/50 border-red-700/40 text-red-300",
    slate:   "bg-slate-900/50 border-slate-700/40 text-slate-400",
  }[color];
  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-semibold text-sm ${cls}`}>
      {children}
    </div>
  );
}

function InfoBox({ label, children, muted }: { label: string; children: React.ReactNode; muted?: boolean }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-slate-500 font-medium mb-1">{label}</div>
      <div className={`text-sm rounded-lg px-3 py-2.5 bg-bg-elevated ${muted ? "text-slate-400 italic" : "text-slate-300"}`}>
        {children}
      </div>
    </div>
  );
}
