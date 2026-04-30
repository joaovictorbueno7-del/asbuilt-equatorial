"""Create tables and seed the initial admin user.
Run from project root with backend on PYTHONPATH:
    cd backend && python -m scripts.init_db
or:
    PYTHONPATH=backend python scripts/init_db.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from app.core.database import init_models, session_scope  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models import Tenant, User, UserRole  # noqa: E402


def slugify(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "tenant"


async def main():
    print(f"[init_db] DATABASE_URL = {settings.DATABASE_URL}")
    await init_models()
    print("[init_db] tables created")

    async with session_scope() as db:
        slug = slugify(settings.ADMIN_TENANT_NAME)
        tenant = (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
        if not tenant:
            tenant = Tenant(name=settings.ADMIN_TENANT_NAME, slug=slug)
            db.add(tenant)
            await db.flush()
            print(f"[init_db] tenant created: {tenant.name} ({tenant.id})")
        else:
            print(f"[init_db] tenant exists: {tenant.name}")

        admin = (await db.execute(select(User).where(User.email == settings.ADMIN_EMAIL.lower()))).scalar_one_or_none()
        if not admin:
            admin = User(
                tenant_id=tenant.id,
                email=settings.ADMIN_EMAIL.lower(),
                password_hash=hash_password(settings.ADMIN_PASSWORD),
                full_name="Admin",
                role=UserRole.ADMIN,
            )
            db.add(admin)
            print(f"[init_db] admin created: {admin.email} (password from .env)")
        else:
            print(f"[init_db] admin exists: {admin.email}")

    print("[init_db] done")


if __name__ == "__main__":
    asyncio.run(main())
