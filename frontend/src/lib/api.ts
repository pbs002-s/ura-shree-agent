/**
 * REST and WebSocket clients for the URA-Shree backend.
 *
 * The socket wrapper exists because a bare WebSocket loses every message sent
 * before it opens and dies silently when the server restarts. This one queues
 * sends until the connection is up and reconnects with a backoff.
 */

import type {
  AppSettings, ModelInfo, ProviderSpec, ProviderState,
  Snapshot, SnapshotDiff, Status, TreeNode,
} from '../types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
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
}

type Listener = (message: Record<string, unknown>) => void

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
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${scheme}://${window.location.host}${this.path}`
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

    ws.onclose = () => {
      this.statusListeners.forEach((fn) => fn(false))
      if (this.closed) return
      // Exponential backoff, capped, so a stopped server does not become a
      // reconnect storm in the browser's network tab.
      const delay = Math.min(500 * 2 ** this.retries, 8000)
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
