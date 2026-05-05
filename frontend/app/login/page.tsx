import LoginForm from "./LoginForm";

export default function LoginPage() {
  return (
    <main className="min-h-screen grid-bg flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-accent to-indigo-500 flex items-center justify-center font-bold text-bg text-lg">
              ⚡
            </div>
            <h1 className="text-3xl font-bold tracking-tight">
              <span className="text-white">OPS AI</span>{" "}
              <span className="text-accent">GRID</span>
            </h1>
          </div>
          <p className="text-sm text-slate-400">
            Plataforma de automação para o setor elétrico
          </p>
        </div>

        <div className="bg-bg-card border border-grid-line rounded-2xl shadow-2xl p-8 backdrop-blur-sm">
          <h2 className="text-xl font-semibold mb-1">Entrar</h2>
          <p className="text-sm text-slate-400 mb-6">
            Acesse sua conta com email e senha
          </p>
          <LoginForm />
        </div>

        <p className="mt-6 text-center text-xs text-slate-500">
          v0.1.0 · Multi-tenant · MFA · Audit log
        </p>
      </div>
    </main>
  );
}
