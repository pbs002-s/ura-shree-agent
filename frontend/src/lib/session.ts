/**
 * Where the frontend gets its base URL, its token and its project.
 *
 * Three things the single-user build could hard-code and a deployed one cannot:
 * the API might not be on the same origin as the page, requests need a bearer
 * token, and every call is scoped to a project. All three live here so no
 * component has to know about any of them.
 *
 * The token is held in memory and mirrored to localStorage. In-memory is what
 * every request reads, so a token refreshed in one tab takes effect without a
 * reload; localStorage is what survives one.
 */

const TOKEN_KEY = 'shree.access_token'
const REFRESH_KEY = 'shree.refresh_token'
const PROJECT_KEY = 'shree.project_id'

/**
 * Empty means "the origin that served this page", which is what the reverse
 * proxy setup wants and what the dev server proxy already handles.
 */
export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

export function socketUrl(path: string): string {
  if (API_BASE_URL) {
    return API_BASE_URL.replace(/^http/, 'ws') + path
  }
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${window.location.host}${path}`
}

let accessToken: string = localStorage.getItem(TOKEN_KEY) ?? ''
let refreshToken: string = localStorage.getItem(REFRESH_KEY) ?? ''
let projectId: string = localStorage.getItem(PROJECT_KEY) ?? 'default'

const expiryListeners = new Set<() => void>()

export const session = {
  get token(): string {
    return accessToken
  },

  get project(): string {
    return projectId
  },

  get authenticated(): boolean {
    return Boolean(accessToken)
  },

  setTokens(access: string, refresh?: string): void {
    accessToken = access
    localStorage.setItem(TOKEN_KEY, access)
    if (refresh !== undefined) {
      refreshToken = refresh
      localStorage.setItem(REFRESH_KEY, refresh)
    }
  },

  setProject(id: string): void {
    projectId = id
    localStorage.setItem(PROJECT_KEY, id)
  },

  clear(): void {
    accessToken = ''
    refreshToken = ''
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(REFRESH_KEY)
  },

  /** Headers every authenticated call carries. */
  headers(): Record<string, string> {
    const headers: Record<string, string> = { 'X-Project-Id': projectId }
    if (accessToken) headers.Authorization = `Bearer ${accessToken}`
    return headers
  },

  /**
   * A WebSocket handshake cannot set headers, so the token rides in the query
   * string on that one path and nowhere else.
   */
  socketQuery(): string {
    const params = new URLSearchParams({ project: projectId })
    if (accessToken) params.set('token', accessToken)
    return params.toString()
  },

  /** Notifies the app that the session is over and a login is needed. */
  expire(): void {
    session.clear()
    expiryListeners.forEach((fn) => fn())
  },

  onExpire(fn: () => void): () => void {
    expiryListeners.add(fn)
    return () => expiryListeners.delete(fn)
  },

  /**
   * Trades the refresh token for a new access token.
   *
   * Returns false when there is nothing to refresh with or the server refuses,
   * which is the caller's signal to send the user back to a login screen
   * rather than retrying.
   */
  async refresh(): Promise<boolean> {
    if (!refreshToken) return false
    try {
      const response = await fetch(apiUrl('/api/auth/refresh'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
      if (!response.ok) return false
      const body = await response.json()
      if (!body?.access_token) return false
      session.setTokens(body.access_token)
      return true
    } catch {
      return false
    }
  },
}

/**
 * OAuth hands the token back in the URL fragment, which never reaches the
 * server and never lands in an access log. Consume it once and clean the bar.
 */
export function consumeOAuthFragment(): boolean {
  if (!window.location.hash.includes('access_token=')) return false
  const params = new URLSearchParams(window.location.hash.slice(1))
  const token = params.get('access_token')
  if (!token) return false
  session.setTokens(token)
  window.history.replaceState(null, '', window.location.pathname + window.location.search)
  return true
}
