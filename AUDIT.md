# AUDIT.md — OPS AI GRID
**Data:** 2026-05-12  
**Auditado por:** Claude Sonnet 4.6  
**Projeto:** `C:\Users\FILIPE-PSE\.claude\agentes de medição\`

---

## 1. BANCO DE DADOS (`ops_ai_grid.db`)

### 1.1 Tabelas Existentes

```sql
SELECT name FROM sqlite_master WHERE type='table'
```

| # | Tabela |
|---|--------|
| 1 | tenants |
| 2 | users |
| 3 | refresh_tokens |
| 4 | audit_logs |
| 5 | pipeline_runs |
| 6 | technical_norms |
| 7 | agent_runs |
| 8 | norm_structures |
| 9 | learning_cases |
| 10 | norm_materials |

**Nota:** A tabela `work_orders` referenciada no `CONTEXT.md` NÃO existe no banco. O campo correto é `pipeline_runs`.

### 1.2 Contagem de Registros

| Tabela | Contagem | Observação |
|--------|----------|------------|
| `tenants` | **1** | "OPS AI GRID HQ" (slug: ops-ai-grid-hq) |
| `users` | **1** | 1 usuário admin registrado |
| `technical_norms` (ativas) | **14** | 14 normas com `ativa=1` |
| `technical_norms` (total) | **14** | Todas ativas |
| `norm_structures` | **197** | Estruturas indexadas (total) |
| `norm_materials` | **2.085** | Materiais indexados (total) |
| `learning_cases` | **0** | Nenhum caso de aprendizado ainda |
| `pipeline_runs` | **0** | Nenhum pipeline executado ainda |
| `agent_runs` | *(não consultado separadamente)* | — |
| `audit_logs` | **21** | Logs de auditoria |
| `refresh_tokens` | **7** | Tokens ativos/expirados |

### 1.3 Norma NT.00022 (única indexada com Vision)

```sql
SELECT processing_status, processing_progress, pages_processed
FROM technical_norms WHERE id='d06b92ba-37c9-4c45-8bf1-e41633674559'
```

| Campo | Valor |
|-------|-------|
| `processing_status` | `done` |
| `processing_progress` | `100%` |
| `pages_processed` | `93` |
| `norm_structures` (para esta norma) | **186** |
| `norm_materials` (para esta norma) | **2.085** |

### 1.4 Lista Completa das Normas Técnicas

| ID | Nome | Status | Páginas Processadas |
|----|------|--------|---------------------|
| `52e52952` | NT.00005.EQTL-05-Criterios-de-Projetos-de-Redes-de-Distribuicao | idle | 0 |
| `7c014a72` | NT.00006.EQTL-04-Padrao-de-Estruturas-de-Rede-de-Distribuicao-Aerea | idle | 0 |
| `4e37b866` | NT 00005 EQTL 05 Criterios Distribuicao | idle | 0 |
| `74ec015a` | NT 00007 EQTL 06 Padrao Equipamentos | idle | 0 |
| `1347ee7c` | NT 00018 EQTL 08 Rede Compacta | idle | 0 |
| `d7273aaa` | NT 00041 EQTL Faixa Servidao | idle | 0 |
| `c73110dc` | NT 00026 EQTL Subestacoes | idle | 0 |
| `f7fe82dd` | NT 00005 EQTL 05 Criterios Distribuicao | idle | 0 |
| `07505447` | NT 00007 EQTL 06 Padrao de Estruturas para Equipamentos | idle | 0 |
| `ac66b25c` | NT 00018 EQTL 08 Rede de Distribuicao Compacta | idle | 0 |
| `d06b92ba` | **NT 00022 EQTL 04 Padrao de Estruturas de Rede Distribuicao Aerea** | **done** | **93** |
| `d14804ab` | NT 00047 EQTL 00 Criterios e Padronizacao de Aterramento | idle | 0 |
| `a3018e70` | nt 00026 eqtl 00 criterios de projetos de subestacoes | idle | 0 |
| `e021be41` | nt 00041 eqtl 02 faixa de servidao para redes e linhas de distribuicao | idle | 0 |

**Observação:** Existe duplicidade de normas — NT.00005, NT.00007, NT.00018, NT.00026 aparecem em múltiplas entradas com nomes ligeiramente diferentes.

### 1.5 Schema das Tabelas Principais

**`agent_runs`:** `tenant_id, user_id, agent_code, status, parent_run_id, pipeline_id, input_payload (JSON), output_payload (JSON), error_message, confidence_score (FLOAT), started_at, finished_at, id, created_at, updated_at`

**`tenants`:** `name, slug, is_active, id, created_at, updated_at`

---

## 2. ARQUIVOS DO PROJETO

### 2.1 Estrutura de Diretórios

```
agentes de medição/
├── ops_ai_grid.db          (3,8 MB) — banco de dados SQLite
├── .env                    (865 B)  — variáveis de ambiente
├── .env.example            (684 B)
├── CONTEXT.md              (4 KB)   — estado atual do projeto
├── README.md               (4,8 KB)
├── test_sample.kmz         (4,4 KB) — KMZ de teste
├── agents/                          — 9 agentes plugáveis
├── backend/                         — FastAPI + serviços
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── api/v1/          — endpoints REST
│   │   ├── auth/            — JWT + MFA
│   │   ├── core/            — config, DB, queue, security
│   │   ├── middleware/      — security headers
│   │   ├── models/          — SQLAlchemy ORM
│   │   └── services/        — lógica de negócio
│   ├── knowledge/
│   │   ├── normas/          — 14 PDFs
│   │   └── pages/           — imagens renderizadas
│   └── scripts/             — utilitários CLI
├── frontend/                        — Next.js 15
│   ├── app/                 — páginas e rotas
│   ├── components/          — componentes reutilizáveis
│   └── lib/                 — utilitários
└── .venv/                           — virtualenv Python
```

### 2.2 Arquivos Python (por tamanho)

| Arquivo (relativo ao projeto) | Linhas | Função |
|-------------------------------|--------|--------|
| `backend/app/api/v1/knowledge.py` | 501 | Endpoints base de normas (upload, reprocess, materiais) |
| `scripts/index_nt00022_targeted.py` | 417 | Indexador dirigido NT.00022 via Vision |
| `backend/app/services/norms_deep_extractor.py` | 350 | Extrator profundo de PDF com PyMuPDF + Vision |
| `agents/kmz_analyzer/vision.py` | 329 | Claude Vision wrapper (comparação campo x norma) |
| `agents/kmz_analyzer/__init__.py` | 323 | Agente 01 — lógica real de análise KMZ |
| `scripts/reprocess_norm.py` | 277 | Reprocessador de normas via CLI |
| `backend/app/services/pipeline.py` | 264 | Orquestrador DAG dos 9 agentes |
| `backend/app/api/v1/auth.py` | 258 | Autenticação JWT + MFA |
| `backend/app/api/v1/agents.py` | 249 | CRUD de agentes e feedback/correção |
| `backend/app/api/v1/dashboard.py` | 244 | Stats e notificações do dashboard |
| `backend/app/services/norm_lookup.py` | 220 | Lookup de estruturas normativas |
| `scripts/index_normas_batch.py` | 192 | Indexação em lote de PDFs |
| `backend/app/api/v1/works.py` | 161 | Endpoint de obras (KMZ upload → pipeline) |
| `backend/app/api/v1/pipelines.py` | 149 | CRUD de pipelines |
| `scripts/migrate_db.py` | 134 | Migração do banco (idempotente) |
| `agents/kmz_analyzer/parser.py` | 113 | Parser KMZ (extração de placemarks e imagens) |
| `backend/app/services/norms_extractor.py` | 104 | Extrator texto + estrutura simples via Claude |
| `scripts/backup_knowledge.py` | 98 | Backup do conhecimento |
| `scripts/backup_project.py` | 95 | Backup do projeto completo |
| `agents/utm_converter/__init__.py` | 69 | Agente 02 — conversão UTM (parcialmente implementado) |
| `backend/app/main.py` | 69 | Entry point FastAPI |

**Total Python:** ~7.441 linhas em 105 arquivos  
**Total TypeScript/TSX:** ~3.865 linhas em 49 arquivos

### 2.3 PDFs em `backend/knowledge/normas/`

| Arquivo PDF | Tamanho |
|-------------|---------|
| `04fc764c35aa_NT.00047.EQTL-00-Criterios-e-Padronizacao-de-Aterramento.pdf` | 2.238 KB |
| `0c6bb02b917d_NT.00006.EQTL-04-Padrao-de-Estruturas...138kV.pdf` | 9.970 KB |
| `1ff22ded4567_NT.00018.EQTL-08-Rede-Compacta.pdf` | **0 KB** — ARQUIVO VAZIO |
| `2a48f7a06035_NT.00007.EQTL-06-Padrao-de-Estruturas-para-Equipamentos.pdf` | 6.654 KB |
| `395a895ad26e_NT.00007.EQTL-06-Padrao-Equipamentos.pdf` | **0 KB** — ARQUIVO VAZIO |
| `42fee01da51d_NT.00018.EQTL-08-Rede-de-Distribuicao-Compacta.pdf` | 13.331 KB |
| `58c18bfd4463_NT.00005.EQTL-05-Criterios-Distribuicao.pdf` | **0 KB** — ARQUIVO VAZIO |
| `87bd3de16967_nt_00026_eqtl_00_criterios_de_projetos_de_subestacoes.pdf` | 2.126 KB |
| `953e5ed5090d_NT.00026.EQTL-Subestacoes.pdf` | **0 KB** — ARQUIVO VAZIO |
| `9b440d0c9f54_NT.00005.EQTL-05-Criterios-de-Projetos-de-Redes-de-Distribuicao.pdf` | 4.227 KB |
| `a1d49923e8fd_NT.00022.EQTL-04-Padrao-de-Estruturas...231-e-345KV.pdf` | 12.313 KB |
| `bc65d0b54d3a_NT.00041.EQTL-Faixa-Servidao.pdf` | **0 KB** — ARQUIVO VAZIO |
| `e4f429d8cec9_NT.00005.EQTL-05-Criterios-Distribuicao.pdf` | **0 KB** — ARQUIVO VAZIO |
| `faf02b23176e_nt_00041_eqtl_02_faixa_de_servidao...distribuicao.pdf` | 1.928 KB |

**Total:** 14 PDFs — **6 estão com 0 bytes** (arquivos corrompidos/vazios)

### 2.4 Imagens em `backend/knowledge/pages/`

- **1 pasta:** `d06b92ba-37c9-4c45-8bf1-e41633674559/` (NT.00022)
- **251 imagens** JPEG renderizadas de páginas do PDF
- Tamanho médio: ~250 KB por imagem

---

## 3. ENDPOINTS BACKEND (FastAPI)

### 3.1 `/api/v1/health`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| GET | `/api/v1/health` | `health_check` | REAL |

### 3.2 `/api/v1/auth`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| POST | `/api/v1/auth/register` | `register` — cria tenant + admin | REAL |
| POST | `/api/v1/auth/login` | `login` — JWT com lockout + MFA | REAL |
| POST | `/api/v1/auth/refresh` | `refresh` — rotate refresh token | REAL |
| POST | `/api/v1/auth/logout` | `logout` — revoga refresh token | REAL |
| POST | `/api/v1/auth/mfa/setup` | `mfa_setup` — gera TOTP secret + QR code | REAL |
| POST | `/api/v1/auth/mfa/enable` | `mfa_enable` — ativa MFA | REAL |
| POST | `/api/v1/auth/mfa/disable` | `mfa_disable` — desativa MFA | REAL |

### 3.3 `/api/v1/users`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| GET | `/api/v1/users/me` | `get_me` — retorna dados do usuário logado | REAL |

### 3.4 `/api/v1/agents`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| GET | `/api/v1/agents` | `get_agents` — lista agentes registrados | REAL |
| POST | `/api/v1/agents/run` | `run_agent` — dispara agente e enfileira execução | REAL |
| GET | `/api/v1/agents/runs/{run_id}` | `get_run` — status de uma execução | REAL |
| GET | `/api/v1/agents/runs` | `list_runs` — lista execuções do tenant | REAL |
| GET | `/api/v1/agents/{agent_code}/cases` | `list_cases` — lista casos de aprendizado | REAL |
| POST | `/api/v1/agents/{agent_code}/cases/{case_id}/correct` | `submit_correction` — correção humana | REAL |
| POST | `/api/v1/agents/{agent_code}/cases/{case_id}/feedback` | `submit_feedback` — feedback binário | REAL |

### 3.5 `/api/v1/works`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| POST | `/api/v1/works` | `create_work` — upload KMZ → dispara kmz_analyzer | REAL |
| GET | `/api/v1/works` | `list_works` — lista obras (AgentRun de kmz_analyzer) | REAL |
| GET | `/api/v1/works/{run_id}` | `get_work` — detalhes de uma obra | REAL |
| GET | `/api/v1/works/{run_id}/image` | `get_work_image` — serve imagem extraída do KMZ | REAL |

### 3.6 `/api/v1/pipelines`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| POST | `/api/v1/pipelines` | `create_pipeline` — upload KMZ → dispara pipeline de 9 agentes | REAL |
| GET | `/api/v1/pipelines` | `list_pipelines` — lista pipelines do tenant | REAL |
| GET | `/api/v1/pipelines/{pipeline_id}` | `get_pipeline` — detalhes + status de todos os agentes | REAL |

### 3.7 `/api/v1/dashboard`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| GET | `/api/v1/dashboard/stats` | `dashboard_stats` — métricas agregadas de pipelines + agentes | REAL |
| GET | `/api/v1/notifications` | `notifications` — notificações derivadas de pipelines | REAL |
| GET | `/api/v1/agents/{code}/stats` | `agent_stats` — stats detalhadas por agente | REAL |

### 3.8 `/api/v1/knowledge`

| Método | Path | Função | Status |
|--------|------|--------|--------|
| GET | `/api/v1/knowledge` | `list_norms` — lista normas com contagem de estruturas | REAL |
| GET | `/api/v1/knowledge/{norm_id}` | `get_norm` — detalhe da norma + estruturas | REAL |
| POST | `/api/v1/knowledge` | `upload_norm` — upload PDF + extração de texto | REAL |
| PATCH | `/api/v1/knowledge/{norm_id}/deactivate` | `deactivate_norm` — soft-disable (nunca delete) | REAL |
| PATCH | `/api/v1/knowledge/{norm_id}/reactivate` | `reactivate_norm` — reativa norma | REAL |
| GET | `/api/v1/knowledge/{norm_id}/pdf` | `download_pdf` — serve PDF original | REAL |
| POST | `/api/v1/knowledge/{norm_id}/reprocess` | `reprocess_norm` — reindexação via Vision em background | REAL |
| GET | `/api/v1/knowledge/{norm_id}/reprocess_status` | `reprocess_status` — progresso do reprocessamento | REAL |
| GET | `/api/v1/knowledge/{norm_id}/materials` | `list_norm_materials` — lista materiais da norma | REAL |
| GET | `/api/v1/knowledge/{norm_id}/page_image` | `page_image` — serve imagem renderizada de página | REAL |

**Total: 31 endpoints** — todos com lógica real implementada (sem stubs no nível de endpoint).

---

## 4. AGENTES

### 4.1 Resumo dos 9 Agentes

| # | Código | Nome | Linhas | Status | Usa Claude? |
|---|--------|------|--------|--------|-------------|
| 01 | `kmz_analyzer` | KMZ Analyzer | 323 + 329 (vision) + 113 (parser) | **REAL** | Sim (Vision) |
| 02 | `utm_converter` | UTM Converter | 69 | **PARCIAL/STUB** | Não |
| 03 | `adherence_tester` | Adherence Tester | 40 | **STUB** | Não |
| 04 | `description_filler` | Description Filler | 48 | **STUB** | Não |
| 05 | `report_generator` | Report Generator | 30 | **STUB** | Não |
| 06 | `anti_reprova` | Anti-Reprova | 46 | **STUB** | Não |
| 07 | `pipeline_supervisor` | Pipeline Supervisor | 38 | **STUB** | Não |
| 08 | `rpa_screen_learner` | RPA Screen Learner | 33 | **STUB** | Não |
| 09 | `master_supervisor` | Master Supervisor | 64 | **PARCIAL/STUB** | Não |

### 4.2 Agente 01 — KMZ Analyzer (`agents/kmz_analyzer/`)

**Status: REAL — Lógica totalmente implementada**

**Fluxo:**
1. `parse_kmz()` — extrai placemarks + imagens do arquivo KMZ (ZIP com KML + fotos)
2. `norm_lookup.enrich_with_norm()` — extrai códigos dos placemarks (ex: "SI3 N1"), normaliza variantes (SI3↔S3I), busca no SQLite
3. Se encontrado na norma: `compare_with_norm()` — envia foto do campo + desenho técnico para Claude Vision (`claude-sonnet-4-5`), retorna JSON de conformidade
4. Se não encontrado: fallback `analyze_image()` — análise básica sem contexto normativo
5. Agrega resultados, calcula score de qualidade, gera casos de aprendizado

**Saída:**
```json
{
  "structures": [...],
  "non_conformities": [...],
  "quality_score": 0-100,
  "image_count": N,
  "total_conformes": N,
  "total_divergentes": N
}
```

**Qualidade técnica:**
- Semáforo asyncio (`MAX_PARALLEL_VISION = 3`) para rate limit
- Normalização de imagem (resize para 1568px, JPEG 85%)
- Parsing robusto de JSON com regex fallback
- Enriquecimento de campos legacy para compatibilidade downstream
- `requires_human_review` quando `confidence < 0.70` ou divergências > 0

### 4.3 Agente 02 — UTM Converter (`agents/utm_converter/`)

**Status: PARCIAL — Conversão UTM implementada matematicamente, mas incompleta**

- Implementa conversão WGS84→UTM via fórmulas matemáticas brutas (sem `pyproj`)
- **Pendente:** conversão policônica (SAD69/SIRGAS) — requer `pyproj` + grid
- **Pendente:** exportação para planilha Excel
- **Pendente:** validação de área de concessão
- Retorna `stub: True` e `excel_path: None`

### 4.4 Agente 03 — Adherence Tester (`agents/adherence_tester/`)

**Status: STUB — Heurística simples hardcoded**

- Conta estruturas por tipo via `Counter`
- Regra hardcoded: "transformador sem para-raios próximos"
- Regra hardcoded: "postes sem cruzeta visível"
- **Não** carrega base UP × Serviço
- Retorna `rules_base_loaded: False, stub: True`

### 4.5 Agente 04 — Description Filler (`agents/description_filler/`)

**Status: STUB — Templates fixos por tipo de estrutura**

- Tem dicionário hardcoded `DEFAULT_DESC` com 9 tipos de estrutura
- Preenche `details` vazio com descrição template
- Não usa IA, não consulta norma
- Retorna `stub: True, description_filled_by_stub: True`

### 4.6 Agente 05 — Report Generator (`agents/report_generator/`)

**Status: STUB — Não gera nenhum arquivo**

- Retorna `report_path_docx: None, report_path_pdf: None`
- Sempre retorna `needs_human: True`
- `CONTEXT.md` identifica este como próximo passo a implementar
- Requer `python-docx` que **não está** em `requirements.txt`

### 4.7 Agente 06 — Anti-Reprova (`agents/anti_reprova/`)

**Status: STUB — Validações básicas hardcoded**

- Verifica placemark sem nome → bloqueio
- Verifica `structure_type = "outro"` → aviso
- Verifica `description_filled_by_stub` → aviso
- Lógica heurística, sem IA

### 4.8 Agente 07 — Pipeline Supervisor (`agents/pipeline_supervisor/`)

**Status: STUB — Média simples de confiança**

- Calcula média de `confidence_score` dos upstream
- Decisão: `autonomous` se `avg >= 0.70`, senão `human`
- Sem lógica real de supervisão ou análise causal

### 4.9 Agente 08 — RPA Screen Learner (`agents/rpa_screen_learner/`)

**Status: STUB — Não faz nada de real**

- Retorna `mode: "observe"`, `examples_seen: 0`, `accuracy: 0.0`
- Notas explícitas: "RPA Citrix requer integração real com PROJ + computer-use"
- Completamente não implementado

### 4.10 Agente 09 — Master Supervisor (`agents/master_supervisor/`)

**Status: PARCIAL/STUB — Agrega scores mas ROI é estimativa fixa**

- Calcula `overall_score` (média ponderada dos upstream)
- Decide `ready_to_send` se `score >= 75` e sem falhas
- ROI = `n_estruturas × 0.15h` (9 min/estrutura — valor hardcoded)
- Retorna `stub: True`

---

## 5. FRONTEND (Next.js 15)

### 5.1 Rotas e Páginas

| Rota | Arquivo | Linhas | Dados Reais? |
|------|---------|--------|--------------|
| `/login` | `app/login/page.tsx` + `LoginForm.tsx` | 36 + 164 | Sim — POST `/api/auth/login` |
| `/dashboard` | `app/dashboard/page.tsx` | 276 | Sim — GET `/api/dashboard/stats` + `/api/pipelines` |
| `/works/new` | `app/works/new/page.tsx` + `NewWorkForm.tsx` | 29 + 264 | Sim — POST `/api/pipelines` + polling |
| `/works/[id]` | `app/works/[id]/page.tsx` + `ResultView.tsx` + `Map.tsx` | 42 + 302 + 58 | Sim — GET `/api/works/{id}` |
| `/pipelines/[id]` | `app/pipelines/[id]/page.tsx` + `Timeline.tsx` | 37 + 250 | Sim — GET `/api/pipelines/{id}` |
| `/agents` | `app/agents/page.tsx` | 100 | Sim — GET `/api/agents` |
| `/agents/[code]` | `app/agents/[code]/page.tsx` + `CasesList.tsx` | 139 + 144 | Sim — GET `/api/agents/{code}/stats` |
| `/agents/kmz_analyzer/train` | `app/agents/kmz_analyzer/train/page.tsx` + `TrainView.tsx` | 37 + 717 | Sim — GET `/api/agents/{code}/cases` |
| `/knowledge` | `app/knowledge/page.tsx` | 124 | Sim — GET `/api/knowledge` |
| `/knowledge/upload` | `app/knowledge/upload/page.tsx` + `UploadForm.tsx` | 27 + 300 | Sim — POST `/api/knowledge` |
| `/knowledge/[id]` | `app/knowledge/[id]/page.tsx` + `NormActions.tsx` | 129 + 69 | Sim — GET `/api/knowledge/{id}` |

### 5.2 Rotas API (Next.js — proxy para backend)

| Método | Rota Next | Backend chamado |
|--------|-----------|-----------------|
| POST | `/api/auth/login` | `POST /api/v1/auth/login` |
| POST | `/api/auth/logout` | `POST /api/v1/auth/logout` |
| GET | `/api/dashboard/stats` | `GET /api/v1/dashboard/stats` |
| GET | `/api/notifications` | `GET /api/v1/notifications` |
| GET | `/api/agents/[code]/cases` | `GET /api/v1/agents/{code}/cases` |
| POST | `/api/agents/[code]/cases/[caseId]/correct` | `POST /api/v1/agents/{code}/cases/{caseId}/correct` |
| POST | `/api/agents/[code]/cases/[caseId]/feedback` | `POST /api/v1/agents/{code}/cases/{caseId}/feedback` |
| GET | `/api/agents/[code]/stats` | `GET /api/v1/agents/{code}/stats` |
| GET/POST | `/api/knowledge` | `GET/POST /api/v1/knowledge` |
| GET | `/api/knowledge/[id]` | `GET /api/v1/knowledge/{id}` |
| POST | `/api/knowledge/[id]/deactivate` | `PATCH /api/v1/knowledge/{id}/deactivate` |
| POST | `/api/knowledge/[id]/reactivate` | `PATCH /api/v1/knowledge/{id}/reactivate` |
| POST | `/api/knowledge/[id]/reprocess` | `POST /api/v1/knowledge/{id}/reprocess` |
| GET | `/api/knowledge/[id]/reprocess_status` | `GET /api/v1/knowledge/{id}/reprocess_status` |
| POST/GET | `/api/pipelines` | `POST/GET /api/v1/pipelines` |
| GET | `/api/pipelines/[id]` | `GET /api/v1/pipelines/{id}` |
| GET/POST | `/api/works` | `GET/POST /api/v1/works` |
| GET | `/api/works/[id]` | `GET /api/v1/works/{id}` |
| GET | `/api/works/[id]/image` | `GET /api/v1/works/{id}/image` |

### 5.3 Componentes

| Componente | Arquivo | Linhas | Função |
|-----------|---------|--------|--------|
| `Header` | `components/Header.tsx` | 54 | Navbar com link para dashboard e logout |
| `NotificationBell` | `components/NotificationBell.tsx` | 124 | Sininho com polling de notificações |
| `Sparkline` | `components/Sparkline.tsx` | 41 | Gráfico de linha semanal no dashboard |

### 5.4 Observações sobre o Frontend

- **Nenhuma dado mock no frontend** — todas as páginas chamam a API real via proxy Next.js
- A autenticação usa cookies `HttpOnly` (`access_token`, `refresh_token`)
- `backendAuthed()` em `lib/api.ts` injeta o Bearer token automaticamente
- `NewWorkForm.tsx` usa polling a cada 2 segundos para acompanhar progresso do pipeline
- `TrainView.tsx` (717 linhas) é a tela mais complexa — permite revisão lado a lado de foto de campo + desenho técnico da norma
- `middleware.ts` (18 linhas) — proteção de rotas, redireciona para `/login` se sem cookie

---

## 6. GIT

### 6.1 Últimos 15 Commits

| Hash | Mensagem |
|------|---------|
| `3aaecb5` | docs: cria CONTEXT.md com estado completo do projeto |
| `1a601c0` | feat: indexação dirigida NT.00022 (93 estruturas) + Agente 01 com comparação normativa |
| `d2efd70` | checkpoint: antes da migração do banco |
| `ad1ff2a` | feat: bootstrap OPS AI GRID SaaS — backend FastAPI + 9 agentes plugáveis |
| `bffb2e4` | chore: limpa projeto antigo, recomeco do zero para SaaS OPS AI GRID |
| `b68e122` | feat: agente de analise de planilha Excel + base de regras UP x Servico |
| `cad254c` | feat: memoria persistente Supabase, aprendizado por feedback, normalizacao de queries |
| `04fa39e` | fix: busca com peso 50 para match direto em lista de materiais de estrutura |
| `4390540` | Auto: atualiza base de normas |
| `f59ffb8` | trigger: reprocessar normas com nova extracao de tabelas |
| `fca2f9f` | fix: extracao de tabelas melhorada, busca por siglas (CE3/CE4) mais precisa |
| `948f9ea` | fix: agente1 mais preciso e conciso, busca melhorada para siglas e tabelas |
| `93abd08` | feat: adiciona aba de chat com Agente 1 especialista em normas |
| `9ce6fbb` | Auto: atualiza base de normas |
| `de9877e` | add: PDFs das normas Equatorial |

### 6.2 Estatísticas

| Métrica | Valor |
|---------|-------|
| Total arquivos rastreados pelo git | **162** |
| Linhas Python (excluindo .venv) | **~3.000 úteis** (7.441 com duplicatas e venv) |
| Linhas TypeScript/TSX | **3.865** |
| Branch atual | `claude/determined-austin-d399ca` (worktree) |
| Branch principal | `main` |
| Mudanças não comitadas | Apenas `.claude/` (settings locais) |

---

## 7. PROBLEMAS IDENTIFICADOS

### 7.1 Funções com Dados Hardcoded / Stub

| Arquivo | Linha | Problema |
|---------|-------|---------|
| `agents/adherence_tester/__init__.py` | 17-39 | Lógica de fraude hardcoded, base UP×Serviço não carregada |
| `agents/description_filler/__init__.py` | 4-13 | Descrições fixas por tipo, sem consultar norma |
| `agents/report_generator/__init__.py` | 16-30 | Não gera nenhum arquivo, retorna `report_path=None` |
| `agents/rpa_screen_learner/__init__.py` | 15-33 | Não faz nada, sempre `mode=observe, accuracy=0.0` |
| `agents/utm_converter/__init__.py` | 59-68 | Conversão policônica pendente (`pyproj` ausente) |
| `agents/master_supervisor/__init__.py` | 33-38 | ROI calculado com fator fixo `0.15h/estrutura` |
| `agents/pipeline_supervisor/__init__.py` | 15-37 | Média simples sem análise causal |
| `backend/app/core/queue.py` | 24-28 | `CeleryQueue.enqueue()` lança `NotImplementedError` |

### 7.2 TODOs e FIXMEs Identificados

**50 ocorrências** de `TODO/FIXME/stub/hardcoded/mock/placeholder` nos arquivos fonte:

**Críticos:**
- `agents/report_generator/__init__.py`: "stub: relatório para N estruturas (geração real pendente)" — nenhum DOCX é gerado
- `agents/rpa_screen_learner/__init__.py`: "stub: RPA Citrix requer integração real com PROJ + computer-use"
- `agents/utm_converter/__init__.py`: "Policônico pendente: requer pyproj + grid SAD69/SIRGAS"
- `backend/app/core/queue.py`: "Stub. Implement when QUEUE_BACKEND=celery and Redis is available"

**No worktree (versão anterior mais rudimentar):**
- `agents/kmz_analyzer/__init__.py` no worktree ainda tem `# TODO: parse KMZ, extract images + KML, call Claude Vision per image` — esta versão está desatualizada em relação ao projeto principal

### 7.3 Dependências Faltantes

| Dependência | Necessária para | Presente em requirements.txt? |
|-------------|-----------------|-------------------------------|
| `python-docx` | `report_generator` (gerar DOCX) | **NÃO** |
| `pyproj` | `utm_converter` (conversão policônica SAD69/SIRGAS) | **NÃO** |
| `celery` | `queue.py` (produção com Redis) | Não (mas ok para dev) |
| `redis` | Queue backend produção | Não (mas ok para dev) |
| `Pillow` | `vision.py` (`Image.open`) | Não listado explicitamente (vem via pymupdf) |

### 7.4 Inconsistências Backend × Frontend

| Inconsistência | Detalhe |
|---------------|---------|
| **`work_orders` inexistente** | `CONTEXT.md` menciona tabela `work_orders`, mas ela não existe. A tabela real é `pipeline_runs` |
| **Dois endpoints de upload paralelos** | `/api/v1/works` e `/api/v1/pipelines` aceitam KMZ — `/works` dispara apenas `kmz_analyzer`, `/pipelines` dispara o pipeline completo de 9 agentes. Frontend usa `/pipelines` em `NewWorkForm.tsx`, mas a página `/works/[id]` consulta `/works/{id}` |
| **Endpoint de works no frontend** | `app/api/works/route.ts` existe mas `app/works/page.tsx` (listagem de obras) não existe — navegação direta para `/works` quebraria |
| **`/works/[id]` vs `/pipelines/[id]`** | Resultado de pipeline está em `/pipelines/[id]` mas o formulário redireciona para `/pipelines/{runId}` — consistente |

### 7.5 PDFs com 0 Bytes

6 dos 14 PDFs em `backend/knowledge/normas/` têm **0 bytes**:
- `1ff22ded4567_NT.00018.EQTL-08-Rede-Compacta.pdf`
- `395a895ad26e_NT.00007.EQTL-06-Padrao-Equipamentos.pdf`
- `58c18bfd4463_NT.00005.EQTL-05-Criterios-Distribuicao.pdf`
- `953e5ed5090d_NT.00026.EQTL-Subestacoes.pdf`
- `bc65d0b54d3a_NT.00041.EQTL-Faixa-Servidao.pdf`
- `e4f429d8cec9_NT.00005.EQTL-05-Criterios-Distribuicao.pdf`

Esses PDFs foram registrados no banco como duplicatas (mesma norma em versões diferentes). O endpoint `GET /knowledge/{id}/pdf` retornaria HTTP 410 "PDF original missing on disk" para essas normas.

### 7.6 Duplicatas no Banco de Normas

13 normas distintas para 7 normas reais. Exemplos de duplicatas:
- NT.00005 aparece em 3 entradas (`52e52952`, `4e37b866`, `f7fe82dd`)
- NT.00007 aparece em 2 entradas (`7c014a72`, `07505447`)
- NT.00018 aparece em 2 entradas (`1347ee7c`, `ac66b25c`)
- NT.00026 aparece em 2 entradas (`c73110dc`, `a3018e70`)
- NT.00041 aparece em 2 entradas (`d7273aaa`, `e021be41`)

### 7.7 Worktree Desatualizado

O diretório `.claude/worktrees/determined-austin-d399ca/` contém uma versão mais antiga do projeto com:
- Agentes em versão stub simples (antes da implementação real do Agente 01)
- Arquivos duplicados que inflam a contagem de linhas Python

---

## 8. ARQUITETURA E DEPENDÊNCIAS

### 8.1 Stack Tecnológico

| Camada | Tecnologia | Versão |
|--------|-----------|--------|
| Backend | FastAPI | 0.115.5 |
| ORM | SQLAlchemy (async) | 2.0.36 |
| Banco (dev) | SQLite via aiosqlite | 0.20.0 |
| Banco (prod) | PostgreSQL via asyncpg | 0.30.0 |
| Autenticação | JWT (python-jose) + Argon2 | — |
| MFA | pyotp (TOTP) + qrcode | — |
| IA | Anthropic Claude (`claude-sonnet-4-5`) | SDK 0.40.0 |
| PDF | PyMuPDF + pypdf | 1.24.13 + 5.1.0 |
| Rate limit | slowapi | 0.1.9 |
| Queue | `asyncio.create_task` (InProcess) | — |
| Frontend | Next.js 15 + TypeScript | — |
| Estilos | Tailwind CSS | — |

### 8.2 Configuração de Produção (Suportada mas não ativa)

- `DATABASE_URL` pode apontar para PostgreSQL — código usa `asyncpg` se não for SQLite
- `QUEUE_BACKEND=celery` + Redis suportado mas `CeleryQueue.enqueue()` lança `NotImplementedError`
- Docker Compose em `deploy/docker-compose.yml` disponível

### 8.3 Pipeline DAG (9 Agentes)

```
Camada 0 (paralelos):
  kmz_analyzer ──────────────────────────────────┐
  utm_converter ──────────────────────────────────┤
  adherence_tester ──────────────────────────────┤
                                                  ↓
Camada 1:
  description_filler ← kmz_analyzer              │
                                                  ↓
Camada 2:
  report_generator ← description_filler           │
                                                  ↓
Camada 3 (paralelos):
  anti_reprova ← report_generator                │
  rpa_screen_learner ← report_generator          │
                                                  ↓
Camada 4:
  pipeline_supervisor ← [kmz, descr, report, anti_reprova]
                                                  ↓
Camada 5:
  master_supervisor ← TODOS os 8 anteriores
```

---

## 9. RESUMO EXECUTIVO

### O que está funcionando de verdade

1. **Autenticação completa** — JWT + refresh token + lockout + MFA TOTP (implementação de produção)
2. **Agente 01 (KMZ Analyzer)** — fluxo real com Claude Vision, comparação campo×norma, aprendizado por feedback
3. **Base de normas NT.00022** — 186 estruturas + 2.085 materiais indexados, 251 imagens de página
4. **Pipeline orchestrator** — DAG de 9 agentes com paralelismo, few-shot learning, logging
5. **Frontend completo** — dashboard, upload KMZ, tela de treinamento, base de conhecimento
6. **Sistema de feedback** — correção humana com penalidade no score, few-shot para próximas execuções
7. **Audit log** — 21 eventos registrados (logins, criação de normas, etc.)

### O que é stub/pendente

1. **Agentes 3 a 8** — todos stubs, incluindo o gerador de relatório que é o core do produto
2. **Geração de DOCX** — não implementada, `python-docx` não está nos requirements
3. **Conversão UTM policônica** — fórmulas parciais sem `pyproj`
4. **13 normas** ainda não indexadas com Vision (apenas NT.00022 está)
5. **Base UP×Serviço** — referenciada no Agente 03 mas nunca carregada
6. **Integração Citrix/PROJ** — Agente 08 completamente não implementado

### Métricas Finais

| Métrica | Valor |
|---------|-------|
| Endpoints backend implementados | **31** (100% REAL) |
| Agentes com lógica real | **1 de 9** (Agente 01) |
| Agentes parcialmente implementados | **2 de 9** (Agente 02 UTM, Agente 09 Master) |
| Agentes stub | **6 de 9** |
| Normas indexadas com Vision | **1 de 14** (NT.00022) |
| Estruturas normativas no banco | **197 total** (186 da NT.00022) |
| Materiais no banco | **2.085** (todos da NT.00022) |
| Linhas de código Python (projeto principal) | **~3.000** |
| Linhas de código TypeScript | **3.865** |
| Pipelines executados em produção | **0** |
| Casos de aprendizado coletados | **0** |
