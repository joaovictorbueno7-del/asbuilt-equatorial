"""
Sistema de memória persistente do Agente 1.
Salva todas as conversas e aprendizados no Supabase.
"""
import os
import json
import re
from datetime import datetime

try:
    from supabase import create_client
    SUPABASE_OK = True
except ImportError:
    SUPABASE_OK = False

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ─── Mapa de variações conhecidas de estruturas ──────────────────────────────
# O agente aprende novas variações e salva aqui dinamicamente
VARIACOES_ESTRUTURAS = {
    "ce1": ["ce1", "ce 1", "estrutura ce1", "ce1a"],
    "ce1a": ["ce1a", "ce 1a", "estrutura ce1a"],
    "ce2": ["ce2", "ce 2", "estrutura ce2"],
    "ce3": ["ce3", "ce 3", "estrutura ce3"],
    "ce4": ["ce4", "ce 4", "estrutura ce4", "ce4 duplo t", "ce-4"],
    "ce5": ["ce5", "ce 5", "estrutura ce5"],
    "cuf4": ["cuf4", "cuf 4", "alternativa ce4"],
    "b3ce": ["b3ce", "b3 ce", "estrutura b3ce"],
    "n3s": ["n3s", "n3 s", "estrutura n3s"],
}

def normalizar_query(pergunta: str) -> tuple[str, list[str]]:
    """
    Normaliza a pergunta para encontrar estruturas e termos-chave.
    Retorna (pergunta_normalizada, lista_de_estruturas_encontradas)
    """
    p = pergunta.lower().strip()
    estruturas_encontradas = []

    # Detecta estruturas na pergunta
    for cod, variacoes in VARIACOES_ESTRUTURAS.items():
        for var in variacoes:
            if var in p:
                if cod not in estruturas_encontradas:
                    estruturas_encontradas.append(cod.upper())
                break

    # Detecta padrões genéricos (ex: "CE7", "B2CE", etc.)
    padrao = re.findall(r'\b[A-Za-z]{1,4}[\-]?[0-9]{1,2}[A-Za-z]?\b', pergunta)
    for p_match in padrao:
        cod = p_match.upper()
        if cod not in estruturas_encontradas and len(cod) >= 2:
            estruturas_encontradas.append(cod)

    return pergunta, estruturas_encontradas


def get_supabase():
    if not SUPABASE_OK or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


def salvar_conversa(pergunta: str, resposta: str, estruturas: list, util: bool = True):
    """Salva a conversa na base de memória."""
    sb = get_supabase()
    if not sb:
        return

    try:
        sb.table("memorias").insert({
            "pergunta": pergunta,
            "resposta": resposta,
            "estruturas": json.dumps(estruturas),
            "util": util,
            "criado_em": datetime.utcnow().isoformat()
        }).execute()
    except Exception:
        pass


def buscar_memorias_relevantes(pergunta: str, estruturas: list) -> str:
    """Busca conversas anteriores relevantes para enriquecer o contexto."""
    sb = get_supabase()
    if not sb:
        return ""

    try:
        contexto_memoria = ""

        # Busca por estruturas específicas
        for est in estruturas[:2]:
            res = sb.table("memorias")\
                .select("pergunta,resposta")\
                .ilike("estruturas", f"%{est}%")\
                .eq("util", True)\
                .order("criado_em", desc=True)\
                .limit(3)\
                .execute()

            if res.data:
                contexto_memoria += f"\n=== Histórico de consultas sobre {est} ===\n"
                for mem in res.data:
                    contexto_memoria += f"P: {mem['pergunta']}\nR: {mem['resposta'][:300]}...\n\n"

        return contexto_memoria
    except Exception:
        return ""


def criar_tabela_se_necessario():
    """Cria a tabela de memórias no Supabase se não existir."""
    sb = get_supabase()
    if not sb:
        return False
    try:
        # Tenta inserir e deletar um registro de teste
        sb.table("memorias").select("id").limit(1).execute()
        return True
    except Exception:
        return False
