import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OPS AI GRID",
  description: "SaaS multi-agente para o setor elétrico",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
