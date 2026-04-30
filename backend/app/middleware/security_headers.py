from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        h = response.headers
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "strict-origin-when-cross-origin"
        h["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        h["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        h["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        return response
