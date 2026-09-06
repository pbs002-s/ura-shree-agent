/**
 * The last line of defence in the render tree.
 *
 * Without one of these, a single component throwing during render unmounts the
 * whole application and leaves a blank page - no message, no way back, and a
 * bug report that says only "it went white". React gives no hook-based way to
 * catch that, so this stays a class component.
 */

import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Shown instead of the default panel, for boundaries around a single pane. */
  fallback?: (error: Error, reset: () => void) => ReactNode
  label?: string
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // The console is the only sink the browser build has. A deployment that
    // wants these collected points its own error reporter at window.onerror.
    console.error(`[${this.props.label ?? 'app'}] render failed`, error, info.componentStack)
  }

  private reset = (): void => {
    this.setState({ error: null })
  }

  render(): ReactNode {
    const { error } = this.state
    if (!error) return this.props.children
    if (this.props.fallback) return this.props.fallback(error, this.reset)

    return (
      <div className="error-boundary">
        <h2 className="error-boundary-title">Something broke in the interface</h2>
        <p className="error-boundary-message">{error.message || String(error)}</p>
        <p className="hint">
          Your workspace and history are on the server and are unaffected.
        </p>
        <div className="error-boundary-actions">
          <button type="button" className="btn" onClick={this.reset}>
            Try again
          </button>
          <button type="button" className="btn btn-primary" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      </div>
    )
  }
}
