/**
 * REST and WebSocket clients for the URA-Shree backend.
 *
 * Two things this layer owns that no component should have to think about:
 * where the API lives (same origin in development, `VITE_API_BASE_URL` when the
 * frontend is served separately) and the credentials every call carries.
 *
 * The socket wrapper exists because a bare WebSocket loses every message sent
 * before it opens and dies silently when the server restarts. This one queues
 * sends until the connection is up and reconnects with a jittered backoff.
 */

import type {
  AppSettings, ModelInfo, ProviderSpec, ProviderState,
  Snapshot, SnapshotDiff, Status, TreeNode,
} from '../types'
import { apiUrl, session, socketUrl } from './session'

async function request<T>(path: string, init?: RequestInit, retry = true): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...session.headers(),
      ...(init?.headers ?? {}),
    },
  })

  if (response.status === 401) {
    // One silent refresh, then give up and let the app show the expiry modal.
    // Retrying past that turns an expired session into a redirect loop.
    if (retry && (await session.refresh())) {
      return request<T>(path, init, false)
    }
    session.expire()
    throw new Error('Your session has expired. Sign in again to continue.')
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* the body was not JSON; the status line is all we have */
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const api = {
  status: () => request<Status>('/api/status'),
  settings: () => request<AppSettings>('/api/settings'),
  updateSettings: (section: string, values: Record<string, unknown>) =>
    post<AppSettings>('/api/settings', { section, values }),

  providers: () =>
    request<{ providers: ProviderSpec[]; configured: Record<string, ProviderState> }>('/api/providers'),
  saveKey: (provider: string, apiKey?: string, baseUrl?: string) =>
    post<ProviderState>('/api/providers/key', { provider, api_key: apiKey, base_url: baseUrl }),
  forgetProvider: (provider: string) =>
    request<{ removed: boolean }>(`/api/providers/${provider}`, { method: 'DELETE' }),
  scanModels: (provider: string, apiKey?: string, baseUrl?: string) =>
    post<{ ok: boolean; provider: string; source: string; models: ModelInfo[]; error?: string }>(
      '/api/providers/scan', { provider, api_key: apiKey, base_url: baseUrl, save: true }),
  selectModel: (provider: string, model: string) =>
    post<AppSettings>('/api/providers/select', { provider, model }),

  tree: () => request<{ tree: TreeNode | null; truncated: boolean; workspace?: string | null }>('/api/tree'),
  readFile: (path: string) =>
    request<{ content: string; path: string; total_lines: number }>(
      `/api/file?path=${encodeURIComponent(path)}`),
  writeFile: (path: string, content: string) => post('/api/file', { path, content }),
  deleteFile: (path: string) =>
    request<{ ok: boolean; deleted: string }>(`/api/file?path=${encodeURIComponent(path)}`, { method: 'DELETE' }),
  selectWorkspace: (path: string) => post<{ ok: boolean; workspace: string | null }>('/api/workspace/select', { path }),
  currentWorkspace: () => request<{ workspace: string | null }>('/api/workspace/current'),
  browseWorkspace: () => post<{ ok: boolean; cancelled?: boolean; workspace: string | null }>('/api/workspace/browse', {}),
  browseDirectory: () => post<{ ok: boolean; cancelled?: boolean; path: string | null }>('/api/browse-directory', {}),

  skills: () => request<import('../types').Skill[]>('/api/skills'),
  addSkill: (name: string, description: string, prompt: string) =>
    post<import('../types').Skill>('/api/skills', { name, description, prompt }),
  toggleSkill: (skillId: string, enabled?: boolean) =>
    request<import('../types').Skill>(`/api/skills/${skillId}`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
      headers: { 'Content-Type': 'application/json' },
    }),
  deleteSkill: (skillId: string) => request<{ ok: boolean }>(`/api/skills/${skillId}`, { method: 'DELETE' }),

  upload: (targetDir: string, files: { path: string; content_base64: string }[]) =>
    post<{ written: { path: string; bytes: number }[]; rejected: { path: string; reason: string }[]; count: number }>(
      '/api/upload', { target_dir: targetDir, files }),

  index: () => post('/api/index', {}),
  memory: () => request<{ summary: string; facts: Record<string, string>; decisions: unknown[] }>('/api/memory'),

  timeline: () =>
    request<{ nodes: Snapshot[]; head: string | null; count: number; store_bytes: number }>('/api/timemachine'),
  snapshot: (label: string) => post<Snapshot>('/api/timemachine/snapshot', { label }),
  diff: (fromId: string, toId: string) =>
    request<SnapshotDiff>(`/api/timemachine/diff?from_id=${fromId}&to_id=${toId}`),
  restore: (snapshotId: string, dryRun = false) =>
    post<{ success: boolean; files_written: number; files_deleted: number; new_head: string }>(
      '/api/timemachine/restore', { snapshot_id: snapshotId, dry_run: dryRun }),

  authProviders: () =>
    request<{ password: boolean; oauth: string[]; mode: string }>('/api/auth/providers'),
  login: (email: string, password: string) =>
    post<{ access_token: string; refresh_token: string; role: string }>(
      '/api/auth/login', { email, password }),
  register: (email: string, password: string, displayName = '') =>
    post<{ access_token: string; refresh_token: string; role: string }>(
      '/api/auth/register', { email, password, display_name: displayName }),
  me: () => request<{ id: string; email: string; role: string; mode: string; local: boolean }>(
    '/api/auth/me'),
  oauthStart: (provider: string) =>
    request<{ url: string; state: string }>(`/api/auth/oauth/${provider}/start`),

  projects: () =>
    request<{ projects: { id: string; name: string; role: string }[] }>('/api/projects'),
  createProject: (name: string) =>
    post<{ id: string; name: string; role: string }>('/api/projects', { name }),

  job: (taskId: string) =>
    request<{ task_id: string; state: string; result?: unknown; error?: string }>(
      `/api/jobs/${taskId}`),
}

type Listener = (message: Record<string, unknown>) => void

const MAX_RECONNECT_DELAY_MS = 15_000

export class Socket {
  private ws: WebSocket | null = null
  private queue: string[] = []
  private listeners = new Set<Listener>()
  private statusListeners = new Set<(open: boolean) => void>()
  private retries = 0
  private closed = false
  private timer: number | null = null
  private path: string

  constructor(path: string) {
    this.path = path
    this.connect()
  }

  private url(): string {
    return `${socketUrl(this.path)}?${session.socketQuery()}`
  }

  private connect(): void {
    if (this.closed) return
    const ws = new WebSocket(this.url())
    this.ws = ws

    ws.onopen = () => {
      this.retries = 0
      this.statusListeners.forEach((fn) => fn(true))
      // Anything queued while the socket was down goes out now, in order.
      const pending = this.queue.splice(0)
      pending.forEach((payload) => ws.send(payload))
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data as string)
        this.listeners.forEach((fn) => fn(message))
      } catch {
        /* a frame that is not JSON is not ours */
      }
    }

    ws.onclose = (event) => {
      this.statusListeners.forEach((fn) => fn(false))
      if (this.closed) return
      // 1008 is the server refusing the credentials. Reconnecting cannot fix
      // that and would spin until the tab is closed.
      if (event.code === 1008) {
        session.expire()
        return
      }
      // Exponential backoff, capped, with jitter so a restarted server does not
      // get every disconnected client back in the same millisecond.
      const base = Math.min(500 * 2 ** this.retries, MAX_RECONNECT_DELAY_MS)
      const delay = base / 2 + Math.random() * (base / 2)
      this.retries += 1
      this.timer = window.setTimeout(() => this.connect(), delay)
    }

    ws.onerror = () => ws.close()
  }

  send(message: Record<string, unknown>): void {
    const payload = JSON.stringify(message)
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(payload)
    else this.queue.push(payload)
  }

  onMessage(fn: Listener): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  onStatus(fn: (open: boolean) => void): () => void {
    this.statusListeners.add(fn)
    fn(this.ws?.readyState === WebSocket.OPEN)
    return () => this.statusListeners.delete(fn)
  }

  close(): void {
    this.closed = true
    if (this.timer) window.clearTimeout(this.timer)
    this.ws?.close()
  }
}

export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** index
  return `${value >= 100 || index === 0 ? Math.round(value) : value.toFixed(1)} ${units[index]}`
}

export function formatNumber(value: number): string {
  return value.toLocaleString('en-US')
}

export function relativeTime(epochSeconds: number): string {
  const seconds = Math.max(0, Date.now() / 1000 - epochSeconds)
  if (seconds < 60) return `${Math.floor(seconds)}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}
