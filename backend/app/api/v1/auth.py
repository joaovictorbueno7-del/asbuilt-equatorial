from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.core.security import (
    hash_password,
    verify_password,
    needs_rehash,
    create_access_token,
    create_refresh_token,
    decode_token,
    encrypt,
    decrypt,
)
from app.models import User, UserRole, RefreshToken, Tenant
from app.auth.mfa import (
    generate_totp_secret,
    provisioning_uri,
    qr_code_data_url,
    verify_totp,
    generate_recovery_codes,
)
from app.auth.dependencies import get_current_user, get_client_ip
from app.services.audit import write_audit

router = APIRouter()

MAX_FAILED = 5
LOCK_MINUTES = 15


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)
    mfa_code: str | None = None


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_minutes: int = settings.JWT_ACCESS_TTL_MINUTES
    mfa_required: bool = False


class RefreshIn(BaseModel):
    refresh_token: str


class MFASetupOut(BaseModel):
    secret: str
    otpauth_uri: str
    qr_code_data_url: str
    recovery_codes: list[str]


class MFAEnableIn(BaseModel):
    code: str


class RegisterIn(BaseModel):
    tenant_name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)
    full_name: str = Field(min_length=2, max_length=200)


def _slugify(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80] or "tenant"


@router.post("/register", response_model=TokenOut, status_code=201)
async def register(payload: RegisterIn, request: Request, db: AsyncSession = Depends(get_db)):
    """Self-service registration creates a new tenant + admin user.
    Disable in production by removing this route or gating it behind an invite system."""
    existing = (await db.execute(select(User).where(User.email == payload.email.lower()))).scalar_one_or_none()
    if existing:
        raise HTTPException(409, "Email already registered")

    slug = _slugify(payload.tenant_name)
    if (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none():
        slug = f"{slug}-{int(datetime.now().timestamp())}"

    tenant = Tenant(name=payload.tenant_name, slug=slug)
    db.add(tenant)
    await db.flush()

    user = User(
        tenant_id=tenant.id,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=UserRole.ADMIN,
    )
    db.add(user)
    await db.flush()
    await write_audit(
        db, action="user.register", user_id=user.id, tenant_id=tenant.id,
        ip_address=await get_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )
    await db.commit()
    return await _issue_tokens(db, user, request)


async def _issue_tokens(db: AsyncSession, user: User, request: Request) -> TokenOut:
    claims = {"tenant_id": user.tenant_id, "role": user.role.value, "email": user.email}
    access = create_access_token(user.id, claims)
    refresh, jti = create_refresh_token(user.id, claims)

    db.add(
        RefreshToken(
            user_id=user.id,
            jti=jti,
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TTL_DAYS),
            user_agent=request.headers.get("user-agent", "")[:500],
            ip_address=await get_client_ip(request),
        )
    )
    user.last_login_at = datetime.now(timezone.utc)
    user.failed_login_attempts = 0
    await db.commit()
    return TokenOut(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginIn, request: Request, db: AsyncSession = Depends(get_db)):
    user = (
        await db.execute(select(User).where(User.email == payload.email.lower()))
    ).scalar_one_or_none()
    ip = await get_client_ip(request)
    ua = request.headers.get("user-agent", "")

    if not user or not user.is_active:
        await write_audit(db, action="auth.login_failed", ip_address=ip, user_agent=ua,
                          metadata={"email": payload.email}, success=False, notes="user_not_found_or_inactive")
        await db.commit()
        raise HTTPException(401, "Invalid credentials")

    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        raise HTTPException(423, "Account locked. Try again later.")

    if not verify_password(payload.password, user.password_hash):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= MAX_FAILED:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCK_MINUTES)
        await write_audit(db, action="auth.login_failed", user_id=user.id, tenant_id=user.tenant_id,
                          ip_address=ip, user_agent=ua, success=False, notes="bad_password")
        await db.commit()
        raise HTTPException(401, "Invalid credentials")

    if user.mfa_enabled:
        if not payload.mfa_code:
            return TokenOut(access_token="", refresh_token="", mfa_required=True)
        secret = decrypt(user.mfa_secret_encrypted) if user.mfa_secret_encrypted else ""
        if not verify_totp(secret, payload.mfa_code):
            user.failed_login_attempts += 1
            await write_audit(db, action="auth.mfa_failed", user_id=user.id, tenant_id=user.tenant_id,
                              ip_address=ip, user_agent=ua, success=False)
            await db.commit()
            raise HTTPException(401, "Invalid MFA code")

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    await write_audit(db, action="auth.login", user_id=user.id, tenant_id=user.tenant_id,
                      ip_address=ip, user_agent=ua)
    return await _issue_tokens(db, user, request)


@router.post("/refresh", response_model=TokenOut)
async def refresh(payload: RefreshIn, request: Request, db: AsyncSession = Depends(get_db)):
    try:
        claims = decode_token(payload.refresh_token)
    except ValueError:
        raise HTTPException(401, "Invalid refresh token")
    if claims.get("type") != "refresh":
        raise HTTPException(401, "Wrong token type")

    jti = claims.get("jti")
    rt = (await db.execute(select(RefreshToken).where(RefreshToken.jti == jti))).scalar_one_or_none()
    if not rt or rt.revoked or rt.expires_at < datetime.now(timezone.utc):
        raise HTTPException(401, "Refresh token revoked or expired")

    user = (await db.execute(select(User).where(User.id == claims["sub"]))).scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(401, "User inactive")

    rt.revoked = True
    return await _issue_tokens(db, user, request)


@router.post("/logout", status_code=204)
async def logout(payload: RefreshIn, db: AsyncSession = Depends(get_db),
                 user: User = Depends(get_current_user)):
    try:
        claims = decode_token(payload.refresh_token)
        jti = claims.get("jti")
    except ValueError:
        return
    rt = (await db.execute(select(RefreshToken).where(RefreshToken.jti == jti))).scalar_one_or_none()
    if rt and rt.user_id == user.id:
        rt.revoked = True
        await db.commit()


@router.post("/mfa/setup", response_model=MFASetupOut)
async def mfa_setup(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if user.mfa_enabled:
        raise HTTPException(400, "MFA already enabled. Disable first to re-setup.")
    secret = generate_totp_secret()
    uri = provisioning_uri(secret, user.email)
    codes = generate_recovery_codes()
    user.mfa_secret_encrypted = encrypt(secret)
    user.mfa_recovery_codes_encrypted = encrypt(",".join(codes))
    await db.commit()
    return MFASetupOut(
        secret=secret,
        otpauth_uri=uri,
        qr_code_data_url=qr_code_data_url(uri),
        recovery_codes=codes,
    )


@router.post("/mfa/enable", status_code=204)
async def mfa_enable(payload: MFAEnableIn, request: Request,
                     user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if not user.mfa_secret_encrypted:
        raise HTTPException(400, "Run /mfa/setup first")
    secret = decrypt(user.mfa_secret_encrypted)
    if not verify_totp(secret, payload.code):
        raise HTTPException(400, "Invalid code")
    user.mfa_enabled = True
    await write_audit(db, action="auth.mfa_enabled", user_id=user.id, tenant_id=user.tenant_id,
                      ip_address=await get_client_ip(request),
                      user_agent=request.headers.get("user-agent", ""))
    await db.commit()


@router.post("/mfa/disable", status_code=204)
async def mfa_disable(payload: MFAEnableIn, request: Request,
                      user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if not user.mfa_enabled:
        return
    secret = decrypt(user.mfa_secret_encrypted) if user.mfa_secret_encrypted else ""
    if not verify_totp(secret, payload.code):
        raise HTTPException(400, "Invalid code")
    user.mfa_enabled = False
    user.mfa_secret_encrypted = None
    user.mfa_recovery_codes_encrypted = None
    await write_audit(db, action="auth.mfa_disabled", user_id=user.id, tenant_id=user.tenant_id,
                      ip_address=await get_client_ip(request),
                      user_agent=request.headers.get("user-agent", ""))
    await db.commit()
