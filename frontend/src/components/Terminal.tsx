import { useEffect, useRef, useState } from 'react'
import { Socket } from '../lib/api'
import { IconRefresh, IconTrash } from '../lib/icons'
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
export function TerminalPanel({ session = 'default' }: { session?: string }) {
  const [lines, setLines] = useState<TerminalLine[]>([
    { kind: 'note', text: 'Persistent shell. Working directory and environment carry over between commands.' },
  ])
  const [draft, setDraft] = useState('')
  const [cwd, setCwd] = useState('.')
  const [running, setRunning] = useState(false)
  const [connected, setConnected] = useState(false)

  const socketRef = useRef<Socket | null>(null)
  const outRef = useRef<HTMLDivElement>(null)
  const history = useRef<string[]>([])
  const historyIndex = useRef(-1)

  const push = (line: TerminalLine) =>
    setLines((prev) => {
      const next = [...prev, line]
      return next.length > MAX_LINES ? next.slice(-MAX_LINES) : next
    })

  useEffect(() => {
    const socket = new Socket(`/ws/terminal?session=${encodeURIComponent(session)}`)
    socketRef.current = socket

    const offStatus = socket.onStatus(setConnected)
    const offMessage = socket.onMessage((message) => {
      const type = message.type as string
      if (type === 'ready' || type === 'info') {
        const info = message.info as { relative_cwd?: string; shell?: string } | undefined
        if (info?.relative_cwd) setCwd(info.relative_cwd)
        if (type === 'ready' && info?.shell) push({ kind: 'note', text: `Shell ready: ${info.shell}` })
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

  const submit = () => {
    const command = draft.trim()
    if (!command || running) return
    push({ kind: 'command', text: `${cwd} $ ${command}` })
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

  return (
    <>
      <div className="pane-head">
        <span className="section-label">Terminal</span>
        <span className={`dot ${connected ? (running ? 'dot-warn' : 'dot-ok') : 'dot-danger'}`} />
        <span className="spacer" />
        <span className="faint mono truncate" style={{ maxWidth: 160 }}>{cwd}</span>
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
      </div>

      <div className="term">
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
          {running && <div className="term-line-note">running…</div>}
        </div>

        <div className="term-input">
          <span className="term-prompt">{cwd} $</span>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={connected ? 'Type a command' : 'Connecting…'}
            spellCheck={false}
            autoComplete="off"
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); submit() }
              else if (e.key === 'ArrowUp') { e.preventDefault(); recall(1) }
              else if (e.key === 'ArrowDown') { e.preventDefault(); recall(-1) }
            }}
          />
        </div>
      </div>
    </>
  )
}
