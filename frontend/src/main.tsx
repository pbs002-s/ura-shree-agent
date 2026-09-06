import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { SessionGate } from './components/Session'
import { consumeOAuthFragment } from './lib/session'

// An OAuth redirect lands here with the token in the fragment. Take it before
// anything renders, so the first request already carries credentials.
consumeOAuthFragment()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary label="root">
      <SessionGate>
        <App />
      </SessionGate>
    </ErrorBoundary>
  </StrictMode>,
)
