/**
 * Sign-in, session expiry and loading placeholders.
 *
 * A deployed instance can refuse a request for a reason that is not an error in
 * the usual sense: the token ran out. That needs its own treatment, because the
 * useful response is "sign in again", not "something went wrong" - and because
 * whatever the user had typed should still be there afterwards, which is why
 * the modal sits over the app rather than replacing it.
 *
 * In local mode the server reports `local: true` and none of this renders.
 */

import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { session } from '../lib/session'

interface AuthModes {
  password: boolean
  oauth: string[]
  mode: string
}

/** A grey block standing in for content that has not arrived yet. */
export function Skeleton({ lines = 3, width = '100%' }: { lines?: number; width?: string }) {
  return (
    <div className="skeleton-group" aria-busy="true" aria-live="polite">
      {Array.from({ length: lines }, (_, index) => (
        <div
          key={index}
          className="skeleton-line"
          // Ragged widths read as text loading; identical bars read as a bug.
          style={{ width: index === lines - 1 ? `calc(${width} * 0.6)` : width }}
        />
      ))}
    </div>
  )
}

export function SkeletonTree({ rows = 8 }: { rows?: number }) {
  return (
    <div className="skeleton-group" aria-busy="true">
      {Array.from({ length: rows }, (_, index) => (
        <div
          key={index}
          className="skeleton-line"
          style={{ marginLeft: `${(index % 3) * 12}px`, width: `${60 + ((index * 13) % 35)}%` }}
        />
      ))}
    </div>
  )
}

interface SignInProps {
  /** Rendered as a blocking screen when signed out, or a modal when expired. */
  variant: 'screen' | 'expired'
  onSignedIn: () => void
}

export function SignIn({ variant, onSignedIn }: SignInProps) {
  const [modes, setModes] = useState<AuthModes | null>(null)
  const [registering, setRegistering] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.authProviders().then(setModes).catch(() => setModes({ password: true, oauth: [], mode: 'cloud' }))
  }, [])

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const result = registering
        ? await api.register(email, password)
        : await api.login(email, password)
      session.setTokens(result.access_token, result.refresh_token)
      onSignedIn()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function startOAuth(provider: string): Promise<void> {
    try {
      const { url } = await api.oauthStart(provider)
      window.location.assign(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const expired = variant === 'expired'

  return (
    <div className="overlay">
      <div className="dialog signin-dialog">
        <div className="dialog-head">
          <span className="dialog-title">
            {expired ? 'Session expired' : registering ? 'Create an account' : 'Sign in'}
          </span>
        </div>
        <div className="dialog-body signin-body">
          {expired && (
            <p className="hint signin-note">
              Your session ran out. Signing in again returns you to exactly where you were.
            </p>
          )}

          {modes?.password !== false && (
            <form onSubmit={submit}>
              <div className="field">
                <label htmlFor="signin-email">Email</label>
                <input
                  id="signin-email"
                  type="email"
                  autoComplete="username"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="signin-password">Password</label>
                <input
                  id="signin-password"
                  type="password"
                  autoComplete={registering ? 'new-password' : 'current-password'}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  required
                  minLength={registering ? 12 : undefined}
                />
                {registering && <p className="hint">At least 12 characters.</p>}
              </div>
              {error && <p className="signin-error">{error}</p>}
              <button type="submit" className="btn btn-primary signin-submit" disabled={busy}>
                {busy ? 'Working…' : registering ? 'Create account' : 'Sign in'}
              </button>
            </form>
          )}

          {(modes?.oauth?.length ?? 0) > 0 && (
            <div className="signin-oauth">
              {modes!.oauth.map((provider) => (
                <button
                  key={provider}
                  type="button"
                  className="btn"
                  onClick={() => startOAuth(provider)}
                >
                  Continue with {provider}
                </button>
              ))}
            </div>
          )}

          {!expired && modes?.password !== false && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => { setRegistering(!registering); setError('') }}
            >
              {registering ? 'I already have an account' : 'Create an account'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Watches for expiry and renders the modal.
 *
 * Kept as its own component so the app tree does not re-render on every token
 * event, only this subtree.
 */
export function SessionGate({ children }: { children: React.ReactNode }) {
  const [expired, setExpired] = useState(false)
  const [needsSignIn, setNeedsSignIn] = useState(false)
  const [ready, setReady] = useState(false)

  useEffect(() => session.onExpire(() => setExpired(true)), [])

  useEffect(() => {
    let cancelled = false

    async function bootstrap(): Promise<void> {
      let local = false
      try {
        local = (await api.me()).local
      } catch {
        if (!cancelled) { setNeedsSignIn(!session.authenticated); setReady(true) }
        return
      }
      if (cancelled) return
      if (!local && !session.authenticated) {
        setNeedsSignIn(true)
        setReady(true)
        return
      }

      // Every request is scoped to a project, so one has to be chosen before
      // the app makes its first call. A new account has none; give it one
      // rather than showing an empty picker on the very first visit.
      if (!local) {
        try {
          const { projects } = await api.projects()
          const project = projects[0] ?? (await api.createProject('My workspace'))
          if (!cancelled) session.setProject(project.id)
        } catch {
          /* the app will surface the failure on its own first request */
        }
      }
      if (!cancelled) setReady(true)
    }

    void bootstrap()
    return () => { cancelled = true }
  }, [])

  return (
    <>
      {/* Holding the tree back until a project is chosen avoids a burst of
          requests that would all 400 for want of a project header. */}
      {ready && children}
      {ready === false && <div style={{ padding: 24 }}><Skeleton lines={4} /></div>}
      {expired && <SignIn variant="expired" onSignedIn={() => { setExpired(false); window.location.reload() }} />}
      {!expired && needsSignIn && (
        <SignIn variant="screen" onSignedIn={() => window.location.reload()} />
      )}
    </>
  )
}
