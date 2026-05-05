import { cookies } from "next/headers";

export const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";
export const API_PREFIX = "/api/v1";

export async function backendFetch(path: string, init: RequestInit = {}) {
  return fetch(`${BACKEND_URL}${API_PREFIX}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    cache: "no-store",
  });
}

export async function backendAuthed(path: string, init: RequestInit = {}) {
  const c = await cookies();
  const token = c.get("access_token")?.value;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string> || {}),
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  return fetch(`${BACKEND_URL}${API_PREFIX}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
}
