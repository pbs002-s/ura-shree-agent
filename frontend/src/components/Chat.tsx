import { useEffect, useMemo, useRef, useState } from 'react'
import { Markdown } from '../lib/markdown'
import {
  IconAlert, IconBolt, IconBrain, IconCheck, IconChevronDown, IconChevronRight, IconClose,
  IconFolder, IconPaperclip, IconSend, IconStop, IconUraShreeLogo,
} from '../lib/icons'
import { formatBytes } from '../lib/api'
import type { Attachment, Block, Turn } from '../types'

/** Whether the scroll container is close enough to the bottom to keep pinning. */
function nearBottom(el: HTMLElement, slack = 120): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < slack
}

function ThinkingCard({ text, isStreaming }: { text: string; isStreaming: boolean }) {
  const [open, setOpen] = useState(false)

  const wordCount = useMemo(() => {
    const trimmed = text.trim()
    return trimmed ? trimmed.split(/\s+/).length : 0
  }, [text])

  return (
    <div className={`thinking-card ${open ? 'open' : 'collapsed'}`}>
      <div className="thinking-header" onClick={() => setOpen((prev) => !prev)}>
        <div className="thinking-title">
          <IconBrain size={14} className={`thinking-icon ${isStreaming ? 'pulse' : ''}`} />
          <span>{isStreaming ? 'Thinking…' : 'Thinking Process'}</span>
          {wordCount > 0 && <span className="thinking-badge">{wordCount} words</span>}
        </div>
        <button
          type="button"
          className="thinking-toggle-btn"
          onClick={(e) => {
            e.stopPropagation()
            setOpen((prev) => !prev)
          }}
          title={open ? 'Hide thinking' : 'Open thinking'}
        >
          <span>{open ? 'Hide' : 'Open'}</span>
          {open ? <IconChevronDown size={13} /> : <IconChevronRight size={13} />}
        </button>
      </div>
      {open && (
        <div className="thinking-body">
          <div className="thinking-content">{text}</div>
        </div>
      )}
    </div>
  )
}

function ToolCard({ block, onApprove }: { block: Block; onApprove?: (id: string, approved: boolean) => void }) {
  const [open, setOpen] = useState(false)
  const tool = block.tool!

  const summary = useMemo(() => {
    const args = tool.arguments ?? {}
    const first = args.path ?? args.command ?? args.pattern ?? args.query ?? args.label
    return typeof first === 'string' ? first : ''
  }, [tool.arguments])

  const isPendingApproval = tool.status === 'awaiting_approval'

  const icon =
    isPendingApproval ? <span className="dot dot-warn" />
      : tool.status === 'running' ? <span className="dot dot-live" />
        : tool.status === 'ok' ? <IconCheck size={13} style={{ color: 'var(--ok)' }} />
          : <IconAlert size={13} style={{ color: 'var(--danger)' }} />

  return (
    <div className={`tool ${isPendingApproval ? 'tool-approval-needed' : ''}`}>
      <button className="tool-head" onClick={() => setOpen(!open)}>
        {open ? <IconChevronDown size={13} /> : <IconChevronRight size={13} />}
        {icon}
        <span className="tool-name">{tool.name}</span>
        {summary && <span className="tool-arg truncate">{summary}</span>}
        <span className="spacer" />
        {isPendingApproval && <span className="chip chip-warn">permission needed</span>}
        {tool.mutating && !isPendingApproval && <span className="chip chip-warn">writes</span>}
      </button>

      {isPendingApproval && (
        <div className="tool-approval-card">
          <div className="tool-approval-text">
            <strong>Permission Request:</strong> Shree wants to save or modify <code>{summary || tool.name}</code>. Allow this action?
          </div>
          <div className="tool-approval-actions">
            <button
              className="btn btn-primary btn-sm"
              onClick={(e) => {
                e.stopPropagation()
                onApprove?.(tool.id, true)
              }}
            >
              Approve & Save
            </button>
            <button
              className="btn btn-ghost btn-sm danger"
              onClick={(e) => {
                e.stopPropagation()
                onApprove?.(tool.id, false)
              }}
            >
              Reject
            </button>
          </div>
        </div>
      )}

      {open && (
        <div className="tool-body">
          <pre>{tool.text || (tool.status === 'running' ? 'running…' : isPendingApproval ? 'Waiting for user permission…' : '(no output)')}</pre>
        </div>
      )}
    </div>
  )
}

function TurnView({ turn, onApproveTool }: { turn: Turn; onApproveTool?: (id: string, approved: boolean) => void }) {
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
        if (block.kind === 'tool') return <ToolCard key={key} block={block} onApprove={onApproveTool} />
        if (block.kind === 'thinking') {
          return (
            <ThinkingCard
              key={key}
              text={block.text}
              isStreaming={Boolean(turn.streaming && index === turn.blocks.length - 1)}
            />
          )
        }
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
  'Build a modern interactive web dashboard with React and TypeScript',
  'Create a REST API with FastAPI, data models, and automated tests',
  'Review the workspace files and suggest architectural improvements',
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
  onApproveTool?: (id: string, approved: boolean) => void
  onOpenSkills?: () => void
}

export function Chat({
  turns, busy, connected, modelLabel, attachments,
  onSend, onStop, onAttach, onAttachFolder, onRemoveAttachment, onApproveTool, onOpenSkills,
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
                <IconUraShreeLogo size={28} />
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
            turns.map((turn) => <TurnView key={turn.id} turn={turn} onApproveTool={onApproveTool} />)
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
            placeholder={connected ? `Ask ${modelLabel === 'shree:latest' ? 'Shree' : modelLabel} to build, explain or fix something…` : 'Reconnecting to backend…'}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
          />

          <div className="composer-bar">
            <div className="composer-actions">
              <button className="composer-chip" onClick={onAttach} title="Attach files" type="button">
                <IconPaperclip size={13} /> <span>Files</span>
              </button>
              <button className="composer-chip" onClick={onAttachFolder} title="Add folder to workspace" type="button">
                <IconFolder size={13} /> <span>Folder</span>
              </button>
              {onOpenSkills && (
                <button className="composer-chip" onClick={onOpenSkills} title="Agent Skills & Workflows" type="button">
                  <IconBolt size={13} style={{ color: 'var(--accent)' }} /> <span>Skills</span>
                </button>
              )}
            </div>
            <span className="spacer" />
            <span className="composer-hint">
              {busy ? 'thinking…' : 'Enter ↵ to send'}
            </span>
            {busy ? (
              <button className="btn-send btn-send-stop" onClick={onStop} title="Stop generation" aria-label="Stop generation">
                <IconStop size={14} />
              </button>
            ) : (
              <button
                className={`btn-send${draft.trim() && connected ? ' active' : ''}`}
                onClick={submit}
                disabled={!draft.trim() || !connected}
                title="Send message"
                aria-label="Send message"
              >
                <IconSend size={14} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
