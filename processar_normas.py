"""
Execute este script UMA VEZ no seu PC para processar as normas e salvar no GitHub.
Uso: python processar_normas.py
Coloque os PDFs das normas na pasta 'normas_pdfs' antes de rodar.
"""
import os
import json
import pdfplumber

PASTA_NORMAS = "normas_pdfs"
SAIDA = "normas_base.json"

def read_pdf(caminho: str) -> str:
    texto = ""
    with pdfplumber.open(caminho) as pdf:
        for page in pdf.pages:
            texto += page.extract_text() or ""
    return texto

def processar():
    if not os.path.exists(PASTA_NORMAS):
        os.makedirs(PASTA_NORMAS)
        print(f"Pasta '{PASTA_NORMAS}' criada. Coloque os PDFs lá e rode novamente.")
        return

    pdfs = [f for f in os.listdir(PASTA_NORMAS) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"Nenhum PDF encontrado em '{PASTA_NORMAS}'.")
        return

    # Carrega base existente
    existentes = {}
    if os.path.exists(SAIDA):
        with open(SAIDA, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existentes[item["id"]] = item

    novos = 0
    for nome_pdf in pdfs:
        caminho = os.path.join(PASTA_NORMAS, nome_pdf)
        print(f"Processando: {nome_pdf}...")
        texto = read_pdf(caminho)
        chunks = [texto[i:i+2000] for i in range(0, len(texto), 1800)]
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc_id = f"{nome_pdf}_{i}"
                if doc_id not in existentes:
                    existentes[doc_id] = {
                        "id": doc_id,
                        "fonte": nome_pdf,
                        "texto": chunk
                    }
                    novos += 1

    todos = list(existentes.values())
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Concluído! {len(todos)} trechos salvos ({novos} novos).")
    print(f"Arquivo gerado: {SAIDA}")
    print(f"\nAgora execute:")
    print(f"  git add normas_base.json")
    print(f"  git commit -m \"Atualiza base de normas\"")
    print(f"  git push")

if __name__ == "__main__":
    processar()
