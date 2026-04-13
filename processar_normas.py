"""
Execute este script UMA VEZ no seu PC para processar as normas e salvar no GitHub.
Uso: python processar_normas.py
Coloque os PDFs das normas na pasta 'normas_pdfs' antes de rodar.
"""
import os
import json
import re
import pdfplumber

PASTA_NORMAS = "normas_pdfs"
SAIDA = "normas_base.json"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400

def extrair_pagina(page) -> str:
    """Extrai texto e tabelas de uma página, mantendo contexto."""
    partes = []

    # Texto normal
    texto = page.extract_text()
    if texto:
        partes.append(texto)

    # Tabelas — converte para texto estruturado legível
    try:
        tabelas = page.extract_tables()
        for tabela in tabelas:
            if not tabela:
                continue
            linhas = []
            for row in tabela:
                celulas = [str(c).strip() if c else "" for c in row]
                linhas.append(" | ".join(celulas))
            partes.append("\n".join(linhas))
    except Exception:
        pass

    return "\n".join(partes)

def read_pdf(caminho: str) -> list:
    """Retorna lista de (numero_pagina, texto_pagina)."""
    paginas = []
    with pdfplumber.open(caminho) as pdf:
        for i, page in enumerate(pdf.pages):
            texto = extrair_pagina(page)
            if texto.strip():
                paginas.append((i + 1, texto))
    return paginas

def fazer_chunks(paginas: list, nome_pdf: str) -> list:
    """
    Cria chunks preservando contexto de página.
    Chunks menores com overlap para não perder tabelas inteiras.
    """
    chunks = []

    # Primeira passagem: chunk por página (preserva tabelas completas)
    for num_pag, texto in paginas:
        # Se a página cabe num chunk, mantém inteira
        if len(texto) <= CHUNK_SIZE:
            chunks.append({
                "fonte": nome_pdf,
                "pagina": num_pag,
                "texto": texto
            })
        else:
            # Divide com overlap
            for j in range(0, len(texto), CHUNK_SIZE - CHUNK_OVERLAP):
                trecho = texto[j:j + CHUNK_SIZE]
                if trecho.strip():
                    chunks.append({
                        "fonte": nome_pdf,
                        "pagina": num_pag,
                        "texto": trecho
                    })

    # Segunda passagem: junta páginas consecutivas em pares
    # Isso resolve o caso onde "CE3" está no título de uma página
    # e os materiais estão na próxima
    pares = []
    for i in range(len(paginas) - 1):
        num_a, txt_a = paginas[i]
        num_b, txt_b = paginas[i + 1]
        combo = txt_a[-800:] + "\n" + txt_b[:800]
        if combo.strip():
            pares.append({
                "fonte": nome_pdf,
                "pagina": f"{num_a}-{num_b}",
                "texto": combo
            })

    return chunks + pares

def processar():
    if not os.path.exists(PASTA_NORMAS):
        os.makedirs(PASTA_NORMAS)
        print(f"Pasta '{PASTA_NORMAS}' criada. Coloque os PDFs lá e rode novamente.")
        return

    pdfs = [f for f in os.listdir(PASTA_NORMAS) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"Nenhum PDF encontrado em '{PASTA_NORMAS}'.")
        return

    todos = []
    for nome_pdf in sorted(pdfs):
        caminho = os.path.join(PASTA_NORMAS, nome_pdf)
        print(f"Processando: {nome_pdf}...")
        paginas = read_pdf(caminho)
        chunks = fazer_chunks(paginas, nome_pdf)
        for i, chunk in enumerate(chunks):
            chunk["id"] = f"{nome_pdf}_p{chunk['pagina']}_{i}"
            todos.append(chunk)
        print(f"  → {len(chunks)} trechos extraídos de {len(paginas)} páginas")

    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Concluído! {len(todos)} trechos salvos.")

if __name__ == "__main__":
    processar()
