from datetime import datetime, timedelta, timezone
from typing import Any
import secrets
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError
from cryptography.fernet import Fernet, InvalidToken
from jose import jwt, JWTError
from .config import settings

_hasher = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=2)
_fernet = Fernet(settings.FERNET_KEY.encode() if isinstance(settings.FERNET_KEY, str) else settings.FERNET_KEY)


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, plain)
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    try:
        return _hasher.check_needs_rehash(hashed)
    except InvalidHashError:
        return True


def encrypt(value: str) -> str:
    return _fernet.encrypt(value.encode()).decode()


def decrypt(token: str) -> str:
    try:
        return _fernet.decrypt(token.encode()).decode()
    except InvalidToken:
        raise ValueError("Invalid encrypted token")


def create_token(subject: str, claims: dict[str, Any], expires_delta: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        **claims,
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "type": token_type,
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(subject: str, claims: dict[str, Any]) -> str:
    return create_token(
        subject, claims, timedelta(minutes=settings.JWT_ACCESS_TTL_MINUTES), "access"
    )


def create_refresh_token(subject: str, claims: dict[str, Any]) -> tuple[str, str]:
    token = create_token(
        subject, claims, timedelta(days=settings.JWT_REFRESH_TTL_DAYS), "refresh"
    )
    decoded = jwt.get_unverified_claims(token)
    return token, decoded["jti"]


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}")
