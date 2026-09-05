import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { Markdown } from '../lib/markdown'
import {
  IconAlert, IconBolt, IconBranch, IconBrain, IconBug, IconCheck, IconChevronRight, IconClose,
  IconCode, IconCompass, IconCopy, IconFileText, IconFlask, IconFolder, IconImage,
  IconPaperclip, IconSend, IconShield, IconStop, IconUraShreeLogo,
} from '../lib/icons'
import { formatBytes } from '../lib/api'
import type { Attachment, Block, Turn } from '../types'

/** Whether the scroll container is close enough to the bottom to keep pinning. */
function nearBottom(el: HTMLElement, slack = 120): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < slack
}

const CODE_EXT = /\.(py|js|jsx|ts|tsx|rs|go|java|c|h|cpp|hpp|rb|php|swift|kt|sh|ps1|sql|css|html?|json|ya?ml|toml)$/i
const IMAGE_EXT = /\.(png|jpe?g|gif|webp|svg|bmp|ico)$/i

function fileIcon(name: string): ReactNode {
  if (IMAGE_EXT.test(name)) return <IconImage size={11} />
  if (CODE_EXT.test(name)) return <IconCode size={11} />
  return <IconFileText size={11} />
}

/** Copy button that confirms in place rather than firing a toast across the app. */
function CopyButton({ text, label = 'Copy message' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)

  const copy = () => {
    navigator.clipboard?.writeText(text).then(
      () => {
        setDone(true)
        window.setTimeout(() => setDone(false), 1400)
      },
      () => undefined,
    )
  }

  return (
    <button
      className={`copy-btn${done ? ' done' : ''}`}
      onClick={copy}
      title={label}
      aria-label={label}
      type="button"
    >
      {done ? <IconCheck size={12} /> : <IconCopy size={12} />}
      <span>{done ? 'Copied' : 'Copy'}</span>
    </button>
  )
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
          <span>{isStreaming ? 'Thinking…' : 'Thinking process'}</span>
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
          <IconChevronRight size={13} className={`chevron${open ? ' open' : ''}`} />
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

  const statePill =
    isPendingApproval ? <span className="pill pill-warn"><span className="dot dot-warn" />permission</span>
      : tool.status === 'running' ? <span className="pill pill-live"><span className="dot dot-live" />running</span>
        : tool.status === 'ok' ? <span className="pill pill-ok"><IconCheck size={11} />done</span>
          : <span className="pill pill-danger"><IconAlert size={11} />failed</span>

  return (
    <div className={`tool${isPendingApproval ? ' tool-approval-needed' : ''}${open ? ' open' : ''}`}>
      <button className="tool-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <IconChevronRight size={13} className={`chevron${open ? ' open' : ''}`} />
        <span className="tool-name">{tool.name}</span>
        {summary && <span className="tool-arg truncate">{summary}</span>}
        <span className="spacer" />
        {tool.mutating && !isPendingApproval && <span className="pill pill-warn">writes</span>}
        {statePill}
      </button>

      {isPendingApproval && (
        <div className="tool-approval-card">
          <div className="tool-approval-text">
            <strong>Permission request:</strong> Shree wants to save or modify{' '}
            <code>{summary || tool.name}</code>. Allow this action?
          </div>
          <div className="tool-approval-actions">
            <button
              className="btn btn-primary btn-sm"
              onClick={(e) => {
                e.stopPropagation()
                onApprove?.(tool.id, true)
              }}
            >
              Approve and save
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
          <pre>
            {tool.text || (tool.status === 'running'
              ? 'running…'
              : isPendingApproval ? 'Waiting for user permission…' : '(no output)')}
          </pre>
        </div>
      )}
    </div>
  )
}

function TurnView({ turn, onApproveTool }: { turn: Turn; onApproveTool?: (id: string, approved: boolean) => void }) {
  if (turn.role === 'user') {
    return (
      <div className="turn turn-right">
        <div className="turn-user">
          {turn.text}
          {!!turn.attachments?.length && (
            <div className="attachments" style={{ padding: '8px 0 0' }}>
              {turn.attachments.map((a) => (
                <span key={a.name} className="attachment">
                  {fileIcon(a.name)}
                  <span className="truncate">{a.name}</span>
                  <span className="faint">{formatBytes(a.size)}</span>
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="turn-tools">
          <CopyButton text={turn.text} />
        </div>
      </div>
    )
  }

  // Only the prose is worth copying; tool output already has its own block.
  const plainText = turn.blocks
    .filter((block) => block.kind === 'text')
    .map((block) => block.text)
    .join('\n\n')
    .trim()

  return (
    <div className="turn">
      <div className="turn-meta">
        <span className="role-badge">
          <IconUraShreeLogo size={13} />
          Shree
        </span>
        {turn.meta?.model && <span className="faint mono truncate">{turn.meta.model}</span>}
        {turn.meta?.durationMs != null && (
          <span className="faint mono">{(turn.meta.durationMs / 1000).toFixed(1)}s</span>
        )}
        <span className="spacer" />
        {!turn.streaming && plainText && <CopyButton text={plainText} />}
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

/** One-click starters. Short on purpose: intent for the model, not a script. */
const PRESETS: { id: string; label: string; icon: ReactNode; prompt: string }[] = [
  {
    id: 'explain',
    label: 'Explain codebase and architecture',
    icon: <IconCompass size={13} />,
    prompt: 'Map this workspace: the entry points, how the layers fit together, where state lives, '
      + 'and the three things a new contributor would get wrong first.',
  },
  {
    id: 'tests',
    label: 'Generate unit tests',
    icon: <IconFlask size={13} />,
    prompt: 'Find the code paths with real branching logic and no test coverage, then write unit '
      + 'tests for them using the test framework this project already uses.',
  },
  {
    id: 'perf',
    label: 'Find bugs and optimise performance',
    icon: <IconBug size={13} />,
    prompt: 'Review the workspace for correctness bugs and performance problems. Rank what you find '
      + 'by impact, and show the failing input or the hot path for each one.',
  },
  {
    id: 'diff',
    label: 'Review uncommitted git changes',
    icon: <IconBranch size={13} />,
    prompt: 'Run git status and git diff, then review the uncommitted changes: correctness first, '
      + 'then anything that should be simplified before it lands.',
  },
  {
    id: 'security',
    label: 'Security and vulnerability audit',
    icon: <IconShield size={13} />,
    prompt: 'Audit this workspace for security problems: input validation at trust boundaries, path '
      + 'traversal, injection, secret handling, and unsafe defaults. Cite file and line.',
  },
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
  onOpenPalette?: () => void
}

export function Chat({
  turns, busy, connected, modelLabel, attachments,
  onSend, onStop, onAttach, onAttachFolder, onRemoveAttachment, onApproveTool, onOpenSkills,
  onOpenPalette,
}: ChatProps) {
  const [draft, setDraft] = useState('')
  const [presetsOpen, setPresetsOpen] = useState(false)
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

  const applyPreset = (text: string) => {
    setDraft(text)
    setPresetsOpen(false)
    inputRef.current?.focus()
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
            <div className="chat-welcome">
              <div className="row" style={{ gap: 11, marginBottom: 8 }}>
                <IconUraShreeLogo size={30} />
                <div style={{ fontSize: 'var(--fs-xl)', fontWeight: 600 }}>
                  {'What are we building?'.split(' ').map((word, i) => (
                    <span key={word} className="word" style={{ '--i': i } as CSSProperties}>
                      <span>{word}</span>
                      {' '}
                    </span>
                  ))}
                </div>
              </div>
              <p className="muted" style={{ marginTop: 0, marginBottom: 18, maxWidth: '58ch' }}>
                Shree reads and edits files, runs commands in a persistent shell, and snapshots the
                workspace before every change so anything can be undone.
              </p>

              <div className="section-label" style={{ marginBottom: 8 }}>Quick actions</div>
              <div className="preset-grid">
                {PRESETS.map((preset, i) => (
                  <button
                    key={preset.id}
                    className="preset rise-in"
                    style={{ animationDelay: `${160 + i * 60}ms` }}
                    onClick={() => applyPreset(preset.prompt)}
                  >
                    <span className="preset-icon">{preset.icon}</span>
                    <span className="truncate">{preset.label}</span>
                  </button>
                ))}
              </div>

              {onOpenPalette && (
                <button className="welcome-hint" onClick={onOpenPalette}>
                  <span>Press</span>
                  <kbd className="kbd">Ctrl</kbd>
                  <kbd className="kbd">K</kbd>
                  <span>for the command palette</span>
                </button>
              )}
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
                  {fileIcon(a.path)}
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
            placeholder={connected
              ? `Ask ${modelLabel === 'shree:latest' ? 'Shree' : modelLabel} to build, explain or fix something…`
              : 'Reconnecting to backend…'}
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
              <div className="preset-menu-wrap">
                <button
                  className={`composer-chip${presetsOpen ? ' active' : ''}`}
                  onClick={() => setPresetsOpen((open) => !open)}
                  title="Prompt presets"
                  type="button"
                >
                  <IconBolt size={13} style={{ color: 'var(--accent)' }} /> <span>Quick actions</span>
                </button>
                {presetsOpen && (
                  <>
                    <div className="preset-menu-scrim" onClick={() => setPresetsOpen(false)} />
                    <div className="preset-menu">
                      {PRESETS.map((preset) => (
                        <button key={preset.id} onClick={() => applyPreset(preset.prompt)}>
                          <span className="preset-icon">{preset.icon}</span>
                          <span className="truncate">{preset.label}</span>
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
              <button className="composer-chip" onClick={onAttach} title="Attach files" type="button">
                <IconPaperclip size={13} /> <span>Files</span>
              </button>
              <button className="composer-chip" onClick={onAttachFolder} title="Add folder to workspace" type="button">
                <IconFolder size={13} /> <span>Folder</span>
              </button>
              {onOpenSkills && (
                <button className="composer-chip" onClick={onOpenSkills} title="Agent skills and workflows" type="button">
                  <IconBrain size={13} /> <span>Skills</span>
                </button>
              )}
            </div>
            <span className="spacer" />
            <span className="composer-hint">
              {busy ? (
                <span className="row" style={{ gap: 6 }}><span className="dot dot-live" />working…</span>
              ) : (
                <>
                  <kbd className="kbd">↵</kbd> send
                  <span className="hint-sep">·</span>
                  <kbd className="kbd">⇧ ↵</kbd> newline
                </>
              )}
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
