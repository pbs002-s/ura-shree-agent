import { useEffect, useRef, useState } from 'react'
import { api, Socket } from '../lib/api'
import { IconClose, IconFolder, IconRefresh, IconTrash } from '../lib/icons'
import type { TerminalLine } from '../types'

const MAX_LINES = 4000

/**
 * The terminal panel.
 *
 * Backed by a shell process that outlives each command, so `cd`, activated
 * virtualenvs and exported variables persist exactly as they would in a real
 * terminal. Output streams in as it is produced rather than appearing all at
 * once when the command exits.
 */
export function TerminalPanel({
  session = 'default',
  workspace,
  onClose,
}: {
  session?: string
  workspace?: string | null
  onClose?: () => void
}) {
  const workspaceBase = workspace ? (workspace.split(/[/\\]/).filter(Boolean).pop() || 'workspace') : 'shree'
  const [lines, setLines] = useState<TerminalLine[]>([
    { kind: 'note', text: 'Persistent shell session. Supports interactive coding agents (claude, aider), REPLs, and system commands.' },
  ])
  const [draft, setDraft] = useState('')
  const [cwd, setCwd] = useState(workspaceBase)
  const [running, setRunning] = useState(false)
  const [connected, setConnected] = useState(false)

  const socketRef = useRef<Socket | null>(null)
  const outRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const history = useRef<string[]>([])
  const historyIndex = useRef(-1)

  const displayCwd = (!cwd || cwd === '.' || cwd.startsWith('.')) ? workspaceBase : cwd

  const push = (line: TerminalLine) =>
    setLines((prev) => {
      const next = [...prev, line]
      return next.length > MAX_LINES ? next.slice(-MAX_LINES) : next
    })

  const [pickingDir, setPickingDir] = useState(false)

  const chooseTerminalDirectory = async () => {
    if (pickingDir) return
    setPickingDir(true)
    try {
      const res = await api.browseDirectory()
      if (res.ok && res.path) {
        submitCommand(`cd "${res.path}"`)
      }
    } catch (err) {
      console.error('Failed to browse terminal directory:', err)
      push({ kind: 'error', text: `Failed to open directory picker: ${(err as Error).message}` })
    } finally {
      setPickingDir(false)
    }
  }

  useEffect(() => {
    const socket = new Socket(`/ws/terminal?session=${encodeURIComponent(session)}`)
    socketRef.current = socket

    const offStatus = socket.onStatus(setConnected)
    const offMessage = socket.onMessage((message) => {
      const type = message.type as string
      if (type === 'ready' || type === 'info') {
        const info = message.info as { relative_cwd?: string; shell?: string } | undefined
        if (info?.relative_cwd) setCwd(info.relative_cwd)
        if (type === 'ready' && info?.shell) push({ kind: 'note', text: `Shell ready (${info.shell})` })
      } else if (type === 'started') {
        setRunning(true)
      } else if (type === 'output') {
        push({ kind: 'output', text: String(message.text ?? '').replace(/\r?\n$/, '') })
      } else if (type === 'exit') {
        setRunning(false)
        if (message.cwd) setCwd(String(message.cwd))
        if (message.error) push({ kind: 'error', text: String(message.error) })
        else if (!message.success) {
          push({ kind: 'error', text: `exit ${message.code} · ${message.duration_ms}ms` })
        } else {
          push({ kind: 'note', text: `exit 0 · ${message.duration_ms}ms` })
        }
      } else if (type === 'error') {
        setRunning(false)
        push({ kind: 'error', text: String(message.message) })
      }
    })

    return () => { offStatus(); offMessage(); socket.close() }
  }, [session])

  useEffect(() => {
    const el = outRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])

  const submitCommand = (cmdText?: string) => {
    const command = (cmdText !== undefined ? cmdText : draft).trim()
    if (!command || running) return
    push({ kind: 'command', text: `${displayCwd} ❯ ${command}` })
    socketRef.current?.send({ action: 'run', command })
    history.current.unshift(command)
    historyIndex.current = -1
    setDraft('')
  }

  const recall = (direction: 1 | -1) => {
    const next = historyIndex.current + direction
    if (next < -1 || next >= history.current.length) return
    historyIndex.current = next
    setDraft(next === -1 ? '' : history.current[next])
  }

  const interrupt = () => {
    socketRef.current?.send({ action: 'interrupt' })
  }

  const QUICK_COMMANDS = [
    { label: 'choose folder…', cmd: '__choose_dir__', desc: 'Browse and choose terminal working directory in File Explorer' },
    { label: 'claude', cmd: 'claude', desc: 'Launch Claude Code interactive window' },
    { label: 'claude -p', cmd: 'claude -p "Explain this project"', desc: 'Run Claude print mode' },
    { label: 'aider', cmd: 'aider', desc: 'Launch Aider AI pair programmer' },
    { label: 'python', cmd: 'python -V', desc: 'Python runtime' },
    { label: 'git status', cmd: 'git status', desc: 'Git status' },
    { label: 'dir', cmd: 'dir', desc: 'List files' },
  ]

  return (
    <>
      <div className="pane-head">
        <span className="section-label">Terminal</span>
        <span className={`dot ${connected ? (running ? 'dot-warn' : 'dot-ok') : 'dot-danger'}`} />
        <span className="spacer" />
        <button
          className="btn btn-ghost btn-sm"
          onClick={chooseTerminalDirectory}
          disabled={pickingDir}
          title={`Working folder: ${cwd}. Click to browse and change location.`}
          style={{ gap: 4, fontSize: 'var(--fs-xs)', padding: '2px 7px', maxWidth: 160 }}
        >
          <IconFolder size={12} style={{ color: 'var(--accent)' }} />
          <span className="mono truncate">{pickingDir ? 'Browsing…' : displayCwd}</span>
        </button>
        <button
          className="btn btn-ghost btn-sm btn-icon"
          onClick={() => setLines([])}
          title="Clear the output"
        >
          <IconTrash size={13} />
        </button>
        <button
          className="btn btn-ghost btn-sm btn-icon"
          onClick={() => socketRef.current?.send({ action: 'restart' })}
          title="Restart the shell"
        >
          <IconRefresh size={13} />
        </button>
        {onClose && (
          <button
            className="btn btn-ghost btn-sm btn-icon"
            onClick={onClose}
            title="Hide terminal"
            aria-label="Hide terminal"
          >
            <IconClose size={13} />
          </button>
        )}
      </div>

      <div className="term">
        <div className="term-quick-chips">
          <span className="term-chip-label">Run:</span>
          {QUICK_COMMANDS.map((qc) => (
            <button
              key={qc.label}
              className="term-chip"
              disabled={qc.cmd === '__choose_dir__' && pickingDir}
              onClick={() => {
                if (qc.cmd === '__choose_dir__') {
                  chooseTerminalDirectory()
                } else if (qc.cmd === 'claude' || qc.cmd === 'aider' || qc.cmd === 'git status' || qc.cmd === 'dir') {
                  submitCommand(qc.cmd)
                } else {
                  setDraft(qc.cmd)
                  inputRef.current?.focus()
                }
              }}
              title={qc.desc}
            >
              {qc.cmd === '__choose_dir__' && pickingDir ? 'opening…' : qc.label}
            </button>
          ))}
        </div>

        <div className="term-out" ref={outRef}>
          {lines.map((line, index) => (
            <div
              key={index}
              className={
                line.kind === 'command' ? 'term-line-cmd'
                  : line.kind === 'error' ? 'term-line-err'
                    : line.kind === 'note' ? 'term-line-note' : undefined
              }
            >
              {line.text}
            </div>
          ))}
          {running && <div className="term-line-note running-pulse">● command running…</div>}
        </div>

        <div className="term-input-bar">
          <div className="term-prompt">
            <span className="term-host">{displayCwd}</span>
            <span className="term-sym">❯</span>
          </div>
          <input
            ref={inputRef}
            className="term-cmd-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={connected ? (running ? 'Command is running…' : 'Run claude, aider, python, git, npm...') : 'Connecting…'}
            spellCheck={false}
            autoComplete="off"
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); submitCommand() }
              else if (e.key === 'ArrowUp') { e.preventDefault(); recall(1) }
              else if (e.key === 'ArrowDown') { e.preventDefault(); recall(-1) }
              else if (e.key === 'c' && e.ctrlKey && running) { e.preventDefault(); interrupt() }
            }}
          />
          {running ? (
            <button
              className="btn btn-sm btn-danger term-btn"
              onClick={interrupt}
              title="Interrupt running command (Ctrl+C)"
            >
              Stop
            </button>
          ) : (
            <button
              className="btn btn-sm btn-primary term-btn"
              onClick={() => submitCommand()}
              disabled={!draft.trim() || !connected}
              title="Run command (Enter)"
            >
              Run
            </button>
          )}
        </div>
      </div>
    </>
  )
}
