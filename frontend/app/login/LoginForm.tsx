"use client";

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";

type Stage = "credentials" | "mfa";

function mapError(status: number, raw: string): string {
  if (status === 401) return "Email ou senha inválidos.";
  if (status === 423) return "Conta bloqueada. Tente novamente em alguns minutos.";
  if (status === 429) return "Muitas tentativas. Aguarde antes de tentar novamente.";
  if (status >= 500) return "Erro no servidor. Tente novamente.";
  return raw || "Não foi possível autenticar.";
}

export default function LoginForm() {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>("credentials");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mfaCode, setMfaCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password,
          mfa_code: stage === "mfa" ? mfaCode : null,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(mapError(res.status, data.error));
        setLoading(false);
        return;
      }
      if (data.mfa_required) {
        setStage("mfa");
        setLoading(false);
        return;
      }
      router.push("/dashboard");
      router.refresh();
    } catch {
      setError("Falha de conexão com o servidor.");
      setLoading(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      {stage === "credentials" && (
        <>
          <Field label="Email">
            <input
              type="email"
              required
              autoFocus
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="input"
              placeholder="voce@empresa.com"
            />
          </Field>
          <Field label="Senha">
            <input
              type="password"
              required
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="input"
              placeholder="••••••••"
            />
          </Field>
        </>
      )}

      {stage === "mfa" && (
        <>
          <div className="rounded-lg bg-bg-elevated border border-grid-line p-3 text-sm text-slate-300">
            Código de autenticação de dois fatores enviado pelo seu app.
          </div>
          <Field label="Código MFA (6 dígitos)">
            <input
              type="text"
              inputMode="numeric"
              pattern="[0-9]{6}"
              maxLength={6}
              required
              autoFocus
              value={mfaCode}
              onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, ""))}
              className="input tracking-[0.5em] text-center text-lg font-mono"
              placeholder="000000"
            />
          </Field>
          <button
            type="button"
            onClick={() => {
              setStage("credentials");
              setMfaCode("");
              setError(null);
            }}
            className="text-xs text-slate-400 hover:text-accent transition"
          >
            ← Voltar
          </button>
        </>
      )}

      {error && (
        <div className="rounded-lg bg-red-950/40 border border-red-900/60 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-accent hover:bg-accent-hover text-bg font-semibold py-2.5 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? "Entrando..." : stage === "mfa" ? "Confirmar código" : "Entrar"}
      </button>

      <style jsx>{`
        :global(.input) {
          width: 100%;
          background-color: #0a0e1a;
          border: 1px solid #1e293b;
          color: #e5e7eb;
          padding: 0.625rem 0.875rem;
          border-radius: 0.5rem;
          outline: none;
          transition: border-color 0.15s, box-shadow 0.15s;
        }
        :global(.input:focus) {
          border-color: #22d3ee;
          box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15);
        }
      `}</style>
    </form>
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
