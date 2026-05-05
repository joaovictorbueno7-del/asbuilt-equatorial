"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function NormActions({ normId, ativa }: { normId: string; ativa: boolean }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [motivo, setMotivo] = useState("");

  async function deactivate() {
    setBusy(true);
    try {
      await fetch(`/api/knowledge/${normId}/deactivate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ motivo }),
      });
      router.refresh();
      setConfirming(false);
    } finally { setBusy(false); }
  }

  async function reactivate() {
    setBusy(true);
    try {
      await fetch(`/api/knowledge/${normId}/reactivate`, { method: "POST" });
      router.refresh();
    } finally { setBusy(false); }
  }

  if (ativa && !confirming) {
    return (
      <button onClick={() => setConfirming(true)}
              className="text-sm px-3 py-1.5 rounded-lg border border-amber-500/40 text-amber-300 hover:bg-amber-500/10">
        🔒 Desativar (preservando)
      </button>
    );
  }

  if (ativa && confirming) {
    return (
      <div className="bg-amber-950/40 border border-amber-900/60 rounded-lg p-3 max-w-md">
        <p className="text-sm text-amber-200 mb-2">
          A norma será marcada como <strong>inativa</strong>. PDF e dados continuam preservados.
          <br /><span className="text-xs text-amber-400/80">Norma nunca é deletada.</span>
        </p>
        <input value={motivo} onChange={(e) => setMotivo(e.target.value)}
               placeholder="motivo (opcional)"
               className="w-full mb-2 bg-bg border border-grid-line rounded px-2 py-1 text-sm" />
        <div className="flex gap-2">
          <button onClick={() => setConfirming(false)} className="text-xs px-3 py-1 rounded text-slate-400 hover:text-white">Cancelar</button>
          <button onClick={deactivate} disabled={busy}
                  className="text-xs px-3 py-1 rounded bg-amber-600 hover:bg-amber-500 text-white">
            {busy ? "Desativando..." : "Confirmar desativação"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <button onClick={reactivate} disabled={busy}
            className="text-sm px-3 py-1.5 rounded-lg border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/10">
      {busy ? "Reativando..." : "↺ Reativar"}
    </button>
  );
}
