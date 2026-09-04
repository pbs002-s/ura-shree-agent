import { useEffect, useMemo, useRef, useState } from 'react'
import { Markdown } from '../lib/markdown'
import {
  IconAlert, IconBolt, IconCheck, IconChevronDown, IconChevronRight, IconClose,
  IconPaperclip, IconSend, IconStop, IconTerminal,
} from '../lib/icons'
import { formatBytes } from '../lib/api'
import type { Attachment, Block, Turn } from '../types'

/** Whether the scroll container is close enough to the bottom to keep pinning. */
function nearBottom(el: HTMLElement, slack = 120): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < slack
}

function ToolCard({ block }: { block: Block }) {
  const [open, setOpen] = useState(false)
  const tool = block.tool!

  const summary = useMemo(() => {
    const args = tool.arguments ?? {}
    const first = args.path ?? args.command ?? args.pattern ?? args.query ?? args.label
    return typeof first === 'string' ? first : ''
  }, [tool.arguments])

  const icon =
    tool.status === 'running' ? <span className="dot dot-live" />
      : tool.status === 'ok' ? <IconCheck size={13} style={{ color: 'var(--ok)' }} />
        : <IconAlert size={13} style={{ color: 'var(--danger)' }} />

  return (
    <div className="tool">
      <button className="tool-head" onClick={() => setOpen(!open)}>
        {open ? <IconChevronDown size={13} /> : <IconChevronRight size={13} />}
        {icon}
        <span className="tool-name">{tool.name}</span>
        {summary && <span className="tool-arg truncate">{summary}</span>}
        <span className="spacer" />
        {tool.mutating && <span className="chip chip-warn">writes</span>}
      </button>
      {open && (
        <div className="tool-body">
          <pre>{tool.text || (tool.status === 'running' ? 'running…' : '(no output)')}</pre>
        </div>
      )}
    </div>
  )
}

function TurnView({ turn }: { turn: Turn }) {
  if (turn.role === 'user') {
    return (
      <div className="turn">
        <div className="turn-user">
          {turn.text}
          {!!turn.attachments?.length && (
            <div className="attachments" style={{ padding: '8px 0 0' }}>
              {turn.attachments.map((a) => (
                <span key={a.name} className="attachment">
                  <IconPaperclip size={11} />
                  <span className="truncate">{a.name}</span>
                  <span className="faint">{formatBytes(a.size)}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="turn">
      <div className="turn-meta">
        <span className="avatar">S</span>
        <span>Shree</span>
        {turn.meta?.model && <span className="faint">· {turn.meta.model}</span>}
        {turn.meta?.durationMs != null && (
          <span className="faint">· {(turn.meta.durationMs / 1000).toFixed(1)}s</span>
        )}
      </div>

      {turn.blocks.map((block, index) => {
        const key = `${turn.id}-${index}`
        if (block.kind === 'tool') return <ToolCard key={key} block={block} />
        if (block.kind === 'thinking') return <div key={key} className="thinking">{block.text}</div>
        if (block.kind === 'error') {
          return (
            <div key={key} className="banner banner-danger">
              <IconAlert size={15} style={{ flex: 'none', marginTop: 1 }} />
              <span>{block.text}</span>
            </div>
          )
        }
        return (
          <div key={key}>
            <Markdown source={block.text} />
            {turn.streaming && index === turn.blocks.length - 1 && <span className="caret" />}
          </div>
        )
      })}

      {turn.streaming && turn.blocks.length === 0 && (
        <div className="row faint"><span className="dot dot-live" /> thinking…</div>
      )}
    </div>
  )
}

const SUGGESTIONS = [
  'Explain how the KV cache in model/model.py works',
  'Run the test suite and summarise the failures',
  'Add type hints to tools/filesystem.py',
  'What changed in the workspace since the last snapshot?',
]

interface ChatProps {
  turns: Turn[]
  busy: boolean
  connected: boolean
  modelLabel: string
  attachments: Attachment[]
  onSend: (text: string) => void
  onStop: () => void
  onAttach: () => void
  onAttachFolder: () => void
  onRemoveAttachment: (name: string) => void
}

export function Chat({
  turns, busy, connected, modelLabel, attachments,
  onSend, onStop, onAttach, onAttachFolder, onRemoveAttachment,
}: ChatProps) {
  const [draft, setDraft] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const pinned = useRef(true)

  // Follow the stream only while the user is already at the bottom. Yanking the
  // view down while someone is reading earlier output is the classic chat bug.
  useEffect(() => {
    const el = scrollRef.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [turns])

  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }, [draft])

  const submit = () => {
    const text = draft.trim()
    if (!text || busy) return
    onSend(text)
    setDraft('')
    pinned.current = true
  }

  return (
    <div className="chat">
      <div
        className="chat-scroll"
        ref={scrollRef}
        onScroll={(e) => { pinned.current = nearBottom(e.currentTarget) }}
      >
        <div className="chat-inner">
          {turns.length === 0 ? (
            <div style={{ paddingTop: 56 }}>
              <div className="row" style={{ gap: 10, marginBottom: 6 }}>
                <span className="avatar" style={{ width: 26, height: 26, fontSize: 12 }}>S</span>
                <div style={{ fontSize: 'var(--fs-xl)', fontWeight: 600 }}>What are we building?</div>
              </div>
              <p className="muted" style={{ marginTop: 0, marginBottom: 20 }}>
                Shree reads and edits files, runs commands in a persistent shell, and snapshots the
                workspace before every change so anything can be undone.
              </p>
              <div style={{ display: 'grid', gap: 6 }}>
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    className="btn"
                    style={{ justifyContent: 'flex-start', height: 34, fontWeight: 400 }}
                    onClick={() => { setDraft(s); inputRef.current?.focus() }}
                  >
                    <IconBolt size={13} style={{ color: 'var(--accent)', flex: 'none' }} />
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((turn) => <TurnView key={turn.id} turn={turn} />)
          )}
        </div>
      </div>

      <div className="composer-wrap">
        <div className="composer">
          {attachments.length > 0 && (
            <div className="attachments" style={{ paddingTop: 9 }}>
              {attachments.map((a) => (
                <span key={a.name} className="attachment">
                  <IconPaperclip size={11} />
                  <span className="truncate">{a.path}</span>
                  <span className="faint">{formatBytes(a.size)}</span>
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => onRemoveAttachment(a.name)}
                    aria-label={`Remove ${a.name}`}
                  >
                    <IconClose size={11} />
                  </button>
                </span>
              ))}
            </div>
          )}

          <textarea
            ref={inputRef}
            rows={1}
            value={draft}
            placeholder={connected ? `Ask ${modelLabel} to build, explain or fix something…` : 'Reconnecting to the backend…'}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
          />

          <div className="composer-bar">
            <button className="btn btn-ghost btn-sm" onClick={onAttach} title="Attach files">
              <IconPaperclip size={13} /> Files
            </button>
            <button className="btn btn-ghost btn-sm" onClick={onAttachFolder} title="Add a folder to the workspace">
              <IconTerminal size={13} /> Folder
            </button>
            <span className="spacer" />
            <span className="faint" style={{ fontSize: 'var(--fs-xs)' }}>
              {busy ? 'working…' : 'Enter to send · Shift+Enter for a new line'}
            </span>
            {busy ? (
              <button className="btn btn-sm btn-danger" onClick={onStop}>
                <IconStop size={11} /> Stop
              </button>
            ) : (
              <button
                className="btn btn-sm btn-primary"
                onClick={submit}
                disabled={!draft.trim() || !connected}
              >
                <IconSend size={13} /> Send
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
