# OPS AI GRID

SaaS multi-tenant para automação de processos no setor elétrico/utilities.
Backend FastAPI + 9 agentes plugáveis + auth JWT/MFA + auditoria + multi-tenant.

## Estrutura

```
backend/        FastAPI app (api, auth, models, services, middleware)
agents/         9 agentes plugáveis (registry pattern)
security/       políticas e docs de segurança
integrations/   conectores externos (S3, Citrix, etc.)
tests/          pytest
deploy/         Dockerfile + docker-compose (Postgres + Redis para depois)
docs/           ADRs e arquitetura
scripts/        init_db.py
frontend/       (Next.js — próxima etapa)
```

## Os 9 agentes

| Código | Nome | Consome de |
|---|---|---|
| `kmz_analyzer` | KMZ Analyzer | — |
| `description_filler` | Description Filler | kmz_analyzer |
| `report_generator` | Report Generator | description_filler |
| `anti_reprova` | Anti-Reprova | report_generator |
| `pipeline_supervisor` | Pipeline Supervisor | (monitor 01–04) |
| `utm_converter` | UTM Converter | — |
| `rpa_screen_learner` | RPA Screen Learner | — |
| `adherence_tester` | Adherence Tester | — |
| `master_supervisor` | Master Supervisor | (todos) |

Cada agente é um pacote em `agents/<code>/` com classe que herda de `BaseAgent` e
se registra via `@register`. Adicionar um novo: criar pasta, decorar a classe,
importar em `agents/__init__.py`.

## Rodar localmente (Windows, sem Docker)

### 1. Pré-requisitos
- Python 3.12+ instalado
- Git

### 2. Setup

```bash
cd "C:/Users/FILIPE-PSE/.claude/agentes de medição"
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r backend/requirements.txt
```

### 3. Configurar `.env`

```bash
cp .env.example .env
```

Gerar `SECRET_KEY` e `FERNET_KEY`:

```bash
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python -c "from cryptography.fernet import Fernet; print('FERNET_KEY=' + Fernet.generate_key().decode())"
```

Cole os valores no `.env` substituindo os placeholders. Defina também `ADMIN_PASSWORD`.

### 4. Inicializar banco + admin

```bash
python scripts/init_db.py
```

Cria `ops_ai_grid.db` (SQLite) com tabelas e usuário admin do `.env`.

### 5. Subir API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health

### 6. Testar login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@opsaigrid.local\",\"password\":\"ChangeMe!2026\"}"
```

Retorna `access_token` + `refresh_token`. Use:

```bash
curl http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

curl http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

### 7. Habilitar MFA

```bash
# 1) gera secret + QR (autenticado)
curl -X POST http://localhost:8000/api/v1/auth/mfa/setup \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

# 2) escanear QR no Google Authenticator / Authy
# 3) ativa enviando código atual
curl -X POST http://localhost:8000/api/v1/auth/mfa/enable \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d "{\"code\":\"123456\"}"
```

A partir daí o login exige `mfa_code`.

## Migrar para Postgres + Redis + Celery (depois de instalar Docker)

1. Instale Docker Desktop
2. `cd deploy && docker compose up -d postgres redis`
3. No `.env` troque:
   - `DATABASE_URL=postgresql+asyncpg://opsai:opsai@localhost:5432/opsai`
   - `QUEUE_BACKEND=celery`
   - `REDIS_URL=redis://localhost:6379/0`
4. `python scripts/init_db.py` (recria schema no Postgres)
5. Reiniciar API. **Zero alteração de código.**

A interface `TaskQueue` em `backend/app/core/queue.py` tem o stub `CeleryQueue`
pronto para implementar quando o Redis estiver disponível.

## Segurança implementada

- Argon2id para senhas (rehash automático em parâmetros antigos)
- JWT access (15min) + refresh (7d) com revogação por `jti` no DB
- MFA TOTP (pyotp) com QR code e códigos de recuperação criptografados (Fernet)
- Lockout após 5 tentativas falhas (15min)
- Rate limit por IP (slowapi)
- Headers de segurança (CSP, HSTS, X-Frame-Options, etc.)
- Audit log append-only de eventos de auth/agentes
- Multi-tenant por `tenant_id` em todas as tabelas escopadas
- RBAC com 4 papéis: `admin`, `supervisor`, `operador`, `auditor`
- CORS restrito por env var

## Testes

```bash
PYTHONPATH=backend pytest tests/ -v
```

## Estrutura preparada para escalar

- SQLAlchemy async → troca SQLite↔Postgres via `DATABASE_URL`
- Fila abstrata → InProcess agora, Celery quando Redis subir
- Storage abstrato → local agora, S3 quando bucket criado
- Models com `tenant_id` em todo lugar → multi-tenant pronto
- Registry de agentes → adicionar 10º agente é só criar pasta
