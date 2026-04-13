import streamlit as st
import os
import json
import io
import pdfplumber
from groq import Groq
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions

# ─── Configuração ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise As-built | Equatorial",
    page_icon="⚡",
    layout="wide"
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
CHROMA_PATH = "./normas_db"

# ─── Funções base ────────────────────────────────────────────────────────────────
def get_groq(key=None):
    return Groq(api_key=key or GROQ_API_KEY)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection("normas_equatorial", embedding_function=ef)

def read_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def normas_count():
    try:
        return get_collection().count()
    except:
        return 0

def call_llm(system_prompt: str, user_prompt: str, key=None) -> str:
    client = get_groq(key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=4096
    )
    return response.choices[0].message.content

def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
    return {}

# ─── Carregamento de normas (feito 1 vez) ────────────────────────────────────────
def carregar_normas(pdfs, groq_key):
    col = get_collection()
    for pdf in pdfs:
        texto = read_pdf(pdf)
        chunks = [texto[i:i+2000] for i in range(0, len(texto), 1800)]
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc_id = f"{pdf.name}_{i}"
                try:
                    col.add(
                        documents=[chunk],
                        ids=[doc_id],
                        metadatas=[{"fonte": pdf.name, "chunk": i}]
                    )
                except Exception:
                    pass  # chunk já existe, ignora
    return col.count()

# ─── AGENTE 1 + 2 — Especialistas em Normas e Serviços (consulta à base) ─────────
def agentes_1_2_consultar_normas(asbuilt_data: dict) -> str:
    col = get_collection()
    total = col.count()
    if total == 0:
        return "Nenhuma norma carregada na base."

    termos = " ".join(
        [m.get("descricao", "") for m in asbuilt_data.get("materiais", [])] +
        [s.get("descricao", "") for s in asbuilt_data.get("servicos", [])]
    )
    query = f"normas elétrica rede distribuição materiais serviços {termos}"

    resultados = col.query(query_texts=[query], n_results=min(6, total))
    trechos = resultados.get("documents", [[]])[0]
    fontes  = [m.get("fonte", "") for m in resultados.get("metadatas", [[]])[0]]

    contexto = ""
    for trecho, fonte in zip(trechos, fontes):
        contexto += f"\n--- Fonte: {fonte} ---\n{trecho}\n"
    return contexto

# ─── AGENTE 3 — Leitor de As-built ───────────────────────────────────────────────
def agente3_ler_asbuilt(pdf_bytes: bytes, groq_key: str) -> dict:
    texto = read_pdf(io.BytesIO(pdf_bytes))

    system = """Você é o Agente 3 — Especialista em Leitura de As-built de redes elétricas da Equatorial.
Extraia TODOS os materiais e serviços do documento.
Retorne APENAS JSON válido neste formato exato:
{
  "projeto": "nome do projeto",
  "materiais": [
    {"codigo": "", "descricao": "", "quantidade": 0, "unidade": ""}
  ],
  "servicos": [
    {"codigo": "", "descricao": "", "quantidade": 0, "unidade": ""}
  ],
  "observacoes": ""
}"""
    user = f"Extraia todos os materiais e serviços deste as-built:\n\n{texto[:7000]}"

    resposta = call_llm(system, user, groq_key)
    resultado = parse_json(resposta)

    if not resultado:
        resultado = {
            "projeto": "Não identificado",
            "materiais": [],
            "servicos": [],
            "observacoes": resposta
        }
    return resultado

# ─── AGENTE 4 — Verificador de Aderência ─────────────────────────────────────────
def agente4_verificar(asbuilt_data: dict, normas_ctx: str, groq_key: str) -> dict:
    system = """Você é o Agente 4 — Verificador de Aderência Serviço vs Material.

REGRAS OBRIGATÓRIAS:
1. Cada serviço DEVE ter material correspondente — sem material = REPROVADO
2. Cada material DEVE ter serviço correspondente — material sem uso = REPROVADO
3. Quantidades devem ser coerentes — excesso ou falta = REPROVADO
4. Especificações técnicas devem estar conformes às normas da Equatorial

Retorne APENAS JSON válido:
{
  "aderente": false,
  "total_servicos": 0,
  "total_materiais": 0,
  "nao_conformidades": [
    {
      "tipo": "servico_sem_material | material_sem_servico | quantidade_incompativel | especificacao_incorreta",
      "descricao": "descrição clara do problema",
      "servico_referencia": "serviço envolvido",
      "material_referencia": "material envolvido",
      "norma_violada": "qual norma / artigo",
      "como_corrigir": "instrução clara de correção"
    }
  ],
  "itens_conformes": ["item ok 1", "item ok 2"]
}"""

    user = f"""Verifique a aderência completa:

AS-BUILT EXTRAÍDO:
{json.dumps(asbuilt_data, ensure_ascii=False, indent=2)}

NORMAS APLICÁVEIS:
{normas_ctx[:5000]}"""

    resposta = call_llm(system, user, groq_key)
    resultado = parse_json(resposta)

    if not resultado:
        resultado = {
            "aderente": False,
            "total_servicos": len(asbuilt_data.get("servicos", [])),
            "total_materiais": len(asbuilt_data.get("materiais", [])),
            "nao_conformidades": [],
            "itens_conformes": []
        }
    return resultado

# ─── AGENTE 5 — Supervisor e Relator Final ───────────────────────────────────────
def agente5_relatorio(asbuilt_data: dict, verificacao: dict, normas_ctx: str, groq_key: str) -> dict:
    system = """Você é o Agente 5 — Supervisor Geral e Relator Final.

Você revisa o trabalho de todos os agentes e emite o parecer técnico definitivo.
Para cada não conformidade, detalhe:
- O que está errado (objetivamente)
- Qual norma foi violada (com referência se possível)
- Como corrigir (instruções práticas)

Se houver inconsistências entre agentes, prevalece a análise mais conservadora (mais restritiva).

Retorne APENAS JSON válido:
{
  "aprovado": false,
  "projeto": "nome do projeto",
  "data": "dd/mm/aaaa",
  "resumo_executivo": "resumo em 2-3 frases",
  "total_servicos": 0,
  "total_materiais": 0,
  "nao_conformidades": [
    {
      "descricao": "título curto da não conformidade",
      "norma": "norma / artigo violado",
      "problema": "descrição técnica do problema",
      "correcao": "como corrigir passo a passo"
    }
  ],
  "itens_aprovados": ["item aprovado 1", "item aprovado 2"],
  "parecer_final": "texto completo do parecer técnico"
}"""

    user = f"""Supervise e emita o relatório final:

DADOS DO AS-BUILT (Agente 3):
{json.dumps(asbuilt_data, ensure_ascii=False, indent=2)}

VERIFICAÇÃO DE ADERÊNCIA (Agente 4):
{json.dumps(verificacao, ensure_ascii=False, indent=2)}

NORMAS CONSULTADAS (Agentes 1 e 2):
{normas_ctx[:3000]}

Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}"""

    resposta = call_llm(system, user, groq_key)
    resultado = parse_json(resposta)

    if not resultado:
        resultado = {
            "aprovado": verificacao.get("aderente", False),
            "projeto": asbuilt_data.get("projeto", "N/A"),
            "data": datetime.now().strftime("%d/%m/%Y"),
            "resumo_executivo": "Análise concluída pelos 5 agentes.",
            "total_servicos": verificacao.get("total_servicos", 0),
            "total_materiais": verificacao.get("total_materiais", 0),
            "nao_conformidades": [
                {
                    "descricao": nc.get("descricao", ""),
                    "norma": nc.get("norma_violada", ""),
                    "problema": nc.get("tipo", ""),
                    "correcao": nc.get("como_corrigir", "")
                }
                for nc in verificacao.get("nao_conformidades", [])
            ],
            "itens_aprovados": verificacao.get("itens_conformes", []),
            "parecer_final": ""
        }
    return resultado

# ─── INTERFACE ────────────────────────────────────────────────────────────────────
# Session state
for k, v in [("step", 1), ("report", None), ("asbuilt_bytes", None), ("asbuilt_name", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# Barra lateral — Admin
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Equatorial_Energia_logo.svg/320px-Equatorial_Energia_logo.svg.png", width=180)
    st.markdown("## ⚙️ Administração")
    st.caption("As normas são aprendidas **uma única vez** e ficam gravadas para sempre.")

    n = normas_count()
    if n > 0:
        st.success(f"✅ Base ativa — {n} trechos de normas")
    else:
        st.warning("⚠️ Nenhuma norma carregada ainda")

    with st.expander("📚 Carregar / Atualizar Normas"):
        groq_key_admin = st.text_input("Chave Groq (admin)", type="password",
                                        value=GROQ_API_KEY,
                                        help="Insira a chave Groq para carregar as normas")
        normas_files = st.file_uploader("PDFs das Normas Equatorial",
                                         type="pdf", accept_multiple_files=True)
        if st.button("📥 Carregar na Base", type="primary"):
            if normas_files:
                with st.spinner("Aprendendo normas... pode levar alguns segundos."):
                    total = carregar_normas(normas_files, groq_key_admin)
                st.success(f"✅ Base atualizada! {total} trechos gravados.")
                st.rerun()
            else:
                st.error("Selecione pelo menos 1 PDF de normas.")

    st.divider()
    groq_key_input = st.text_input("🔑 Chave Groq (análise)", type="password",
                                    value=GROQ_API_KEY,
                                    help="Necessária para rodar os 5 agentes")
    if groq_key_input:
        GROQ_API_KEY = groq_key_input

# Cabeçalho
st.title("⚡ Análise de As-built — Equatorial")
st.caption("5 agentes especializados analisam seu documento automaticamente")

# Indicador de etapas
c1, c2, c3 = st.columns(3)
etapas = [("📄 Upload", 1), ("⚙️ Análise dos Agentes", 2), ("📋 Relatório Final", 3)]
for col, (nome, num) in zip([c1, c2, c3], etapas):
    with col:
        if st.session_state.step > num:
            st.success(f"✅ {nome}")
        elif st.session_state.step == num:
            st.info(f"▶️ {nome}")
        else:
            st.markdown(f"<div style='padding:8px;border-radius:6px;background:#f0f0f0;text-align:center'>⚪ {nome}</div>", unsafe_allow_html=True)

st.divider()

# ═══════ ETAPA 1 — Upload ═══════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.header("📄 Etapa 1 — Upload do As-built")

    arquivo = st.file_uploader("Selecione o PDF do As-built", type="pdf")

    pronto = arquivo and bool(GROQ_API_KEY) and normas_count() > 0

    if not GROQ_API_KEY:
        st.error("Configure a chave Groq na barra lateral.")
    elif normas_count() == 0:
        st.warning("Carregue as normas da Equatorial na barra lateral antes de analisar.")
    elif arquivo:
        st.success("Arquivo carregado. Clique em **Iniciar Análise** para continuar.")

    if st.button("▶️ Iniciar Análise", type="primary", disabled=not pronto):
        st.session_state.asbuilt_bytes = arquivo.read()
        st.session_state.asbuilt_name = arquivo.name
        st.session_state.step = 2
        st.rerun()

# ═══════ ETAPA 2 — Processamento ════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.header("⚙️ Etapa 2 — Agentes analisando...")

    barra  = st.progress(0)
    status = st.empty()

    agentes_log = st.container()

    try:
        with agentes_log:
            # Agente 3
            status.markdown("### 🤖 Agente 3 — Lendo o as-built...")
            barra.progress(15, "Agente 3: Extraindo materiais e serviços...")
            asbuilt = agente3_ler_asbuilt(st.session_state.asbuilt_bytes, GROQ_API_KEY)
            st.success(f"✅ Agente 3 — Extraiu {len(asbuilt.get('materiais',[]))} materiais e {len(asbuilt.get('servicos',[]))} serviços")

            # Agentes 1 + 2
            status.markdown("### 🤖 Agentes 1 e 2 — Consultando normas...")
            barra.progress(35, "Agentes 1+2: Consultando base de normas...")
            normas_ctx = agentes_1_2_consultar_normas(asbuilt)
            st.success("✅ Agentes 1 e 2 — Normas consultadas com sucesso")

            # Agente 4
            status.markdown("### 🤖 Agente 4 — Verificando aderência serviço vs material...")
            barra.progress(60, "Agente 4: Verificando aderência...")
            verificacao = agente4_verificar(asbuilt, normas_ctx, GROQ_API_KEY)
            nc = len(verificacao.get("nao_conformidades", []))
            st.success(f"✅ Agente 4 — Encontrou {nc} não conformidade(s)")

            # Agente 5
            status.markdown("### 🤖 Agente 5 — Supervisionando e gerando relatório...")
            barra.progress(85, "Agente 5: Gerando relatório final...")
            relatorio = agente5_relatorio(asbuilt, verificacao, normas_ctx, GROQ_API_KEY)
            st.success("✅ Agente 5 — Relatório final gerado")

        barra.progress(100, "✅ Análise completa!")
        status.markdown("### ✅ Todos os agentes concluíram!")

        st.session_state.report = relatorio
        st.session_state.step = 3
        st.rerun()

    except Exception as e:
        st.error(f"Erro durante a análise: {e}")
        if st.button("⬅️ Voltar e tentar novamente"):
            st.session_state.step = 1
            st.rerun()

# ═══════ ETAPA 3 — Relatório ═════════════════════════════════════════════════════
elif st.session_state.step == 3:
    rel = st.session_state.report

    if rel.get("aprovado"):
        st.success("# ✅ AS-BUILT APROVADO")
    else:
        st.error("# ❌ AS-BUILT REPROVADO")

    col1, col2 = st.columns(2)
    col1.markdown(f"**Projeto:** {rel.get('projeto', 'N/A')}")
    col2.markdown(f"**Data:** {rel.get('data', datetime.now().strftime('%d/%m/%Y'))}")

    if rel.get("resumo_executivo"):
        st.info(f"📝 {rel['resumo_executivo']}")

    # Métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Não Conformidades", len(rel.get("nao_conformidades", [])))
    m2.metric("Itens Aprovados",   len(rel.get("itens_aprovados", [])))
    m3.metric("Serviços",          rel.get("total_servicos", 0))
    m4.metric("Materiais",         rel.get("total_materiais", 0))

    st.divider()

    # Não conformidades
    ncs = rel.get("nao_conformidades", [])
    if ncs:
        st.subheader("❌ Não Conformidades")
        for i, nc in enumerate(ncs, 1):
            with st.expander(f"**{i}. {nc.get('descricao', 'Não conformidade')}**", expanded=(i == 1)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Norma violada:**\n{nc.get('norma', 'N/A')}")
                    st.markdown(f"**Problema:**\n{nc.get('problema', 'N/A')}")
                with col_b:
                    st.markdown(f"**Como corrigir:**\n{nc.get('correcao', 'N/A')}")

    # Itens aprovados
    aprovados = rel.get("itens_aprovados", [])
    if aprovados:
        st.subheader("✅ Itens Aprovados")
        cols = st.columns(2)
        for i, item in enumerate(aprovados):
            cols[i % 2].markdown(f"- {item}")

    # Parecer final
    if rel.get("parecer_final"):
        st.divider()
        st.subheader("📋 Parecer Técnico Final")
        st.markdown(rel["parecer_final"])

    st.divider()

    # Gera texto do relatório para download
    linhas_nc = ""
    for i, nc in enumerate(ncs, 1):
        linhas_nc += f"""
{i}. {nc.get('descricao','')}
   Norma: {nc.get('norma','')}
   Problema: {nc.get('problema','')}
   Correção: {nc.get('correcao','')}
"""

    relatorio_txt = f"""RELATÓRIO DE ANÁLISE DE AS-BUILT — EQUATORIAL
{'='*60}
Status : {'✅ APROVADO' if rel.get('aprovado') else '❌ REPROVADO'}
Projeto: {rel.get('projeto','N/A')}
Data   : {rel.get('data','')}
Arquivo: {st.session_state.asbuilt_name}

RESUMO EXECUTIVO:
{rel.get('resumo_executivo','')}

{'='*60}
NÃO CONFORMIDADES ({len(ncs)}):
{linhas_nc if ncs else 'Nenhuma não conformidade encontrada.'}

{'='*60}
ITENS APROVADOS:
{chr(10).join(['- ' + a for a in aprovados]) if aprovados else 'Nenhum item aprovado listado.'}

{'='*60}
PARECER TÉCNICO FINAL:
{rel.get('parecer_final','')}

{'='*60}
Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Sistema: 5 Agentes IA — Equatorial As-built Analyzer
"""

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🔄 Nova Análise", type="primary", use_container_width=True):
            st.session_state.step   = 1
            st.session_state.report = None
            st.rerun()
    with b2:
        st.download_button(
            "⬇️ Baixar Relatório (.txt)",
            data=relatorio_txt,
            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
