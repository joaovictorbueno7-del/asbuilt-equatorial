# OPS AI GRID — Contexto do Projeto

## O que é
Sistema multi-agente SaaS para o setor elétrico brasileiro. Processa arquivos KMZ
de campo (as-built), valida contra normas técnicas da Equatorial e gera relatórios
de conformidade automáticos.

---

## Estado atual (2026-05-12)

### Base de normas — NT.00022.EQTL-04 ✅ INDEXADA
- **93 estruturas** indexadas com Claude Vision (indexação dirigida por página)
- **1.038 materiais** catalogados com código, descrição, quantidade e tensão
- **251 imagens** de página renderizadas em `backend/knowledge/pages/{norm_id}/`
- Grupos cobertos: BT, MT-MONO, MT-BI, MT-BI-BECO, MT-TRI, MT-TRI-PILAR, EQUIP, TRAFO, ESPECIAL, MUFLA
- Normalização de notação de campo: `SI3→S3I`, `SI4→S4I`, `SI1→S1I` automático

### Outros PDFs (13 restantes)
- Carregados no banco (`ativa=1`), textos extraídos via PyPDF
- **Não** reindexados com Vision ainda
- Podem ser indexados com: `python scripts/reprocess_norm.py --all`

### Agente 01 — KMZ Analyzer ✅ FLUXO REAL
Fluxo implementado:
1. Parse KMZ → placemarks + fotos
2. `norm_lookup.parse_structure_codes()` extrai códigos do nome do placemark
3. `norm_lookup.lookup_structures()` busca no banco com normalização
4. Se encontrado: `vision.compare_with_norm()` — foto do campo + desenho técnico → Claude Vision
5. Resposta: `{conformidade, estruturas_declaradas, estruturas_confirmadas, estruturas_divergentes, materiais_faltantes}`
6. Se não encontrado: fallback para `vision.analyze_image()` (análise básica)

### Tela de Treinamento ✅ ATUALIZADA
- Layout: foto do campo (esq) | desenho técnico da norma (dir)
- Badge de conformidade (verde/vermelho)
- Seletor de estrutura quando há múltiplos desenhos
- Formulário de correção: "Estrutura real = N3" → agente aprende
- Botões: ✅ Confirmado | ❌ Divergência | Pular

---

## Arquitetura

### Pipeline (9 agentes)
```
kmz_analyzer → description_filler → report_generator → anti_reprova → pipeline_supervisor
                    ↗ utm_converter (paralelo)
                    ↗ adherence_tester (paralelo)
                                          → rpa_screen_learner (pós-relatório)
                                                              → master_supervisor (final)
```

### Banco de dados
- SQLite: `ops_ai_grid.db` (→ Postgres em produção via `DATABASE_URL`)
- Tabelas: `tenants`, `users`, `work_orders`, `pipelines`, `agent_runs`, `learning_cases`
- Normas: `technical_norms`, `norm_structures`, `norm_materials`

### Serviços-chave
| Arquivo | Função |
|---------|--------|
| `backend/app/services/norm_lookup.py` | Parse KMZ + lookup norma + carrega desenho |
| `agents/kmz_analyzer/vision.py` | `compare_with_norm()` + `analyze_image()` |
| `backend/app/services/norms_deep_extractor.py` | Indexador genérico (classificação automática) |
| `scripts/index_nt00022_targeted.py` | Indexador dirigido NT.00022 (94 páginas fixas) |

---

## Próximo passo imediato
**Agente 03 — Report Generator** (implementação interrompida)

Deve gerar DOCX no formato exato da Equatorial:
- Cabeçalho: Logo Equatorial (esq) | título centralizado | logo parceira (dir) | linha separadora
- Página 1: Nota, Município, Parceira Construção, título negrito "Postes, Estruturas e Redes"
- Grid: 2 fotos por linha, legenda em negrito: "[Código estrutura] - [tipo poste] [estrutura]"
- Endpoint: `GET /api/v1/works/{id}/report/download`
- Botão download em `/works/{id}`
- Dependência: adicionar `python-docx` em `backend/requirements.txt`

Para testar o fluxo completo: `python scripts/index_nt00022_targeted.py --dry-run`

---

## Comandos úteis
```bash
# Indexar NT.00022 (já feito — usa se precisar reindexar)
python scripts/index_nt00022_targeted.py

# Reprocessar uma norma específica (genérico)
python scripts/reprocess_norm.py <norm_id>

# Reprocessar todas as normas
python scripts/reprocess_norm.py --all

# Migração do banco (idempotente)
python scripts/migrate_db.py

# Backup completo
python scripts/backup_project.py
```
