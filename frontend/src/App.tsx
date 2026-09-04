import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Chat } from './components/Chat'
import { CodeEditorModal } from './components/CodeEditorModal'
import { DiffView } from './components/DiffView'
import { FilesPanel, MemoryPanel, TimelinePanel } from './components/Sidebar'
import { SettingsDialog } from './components/Settings'
import { SkillsModal } from './components/SkillsModal'
import { TerminalPanel } from './components/Terminal'
import { api, formatBytes, Socket } from './lib/api'
import {
  IconAlert, IconBolt, IconBrain, IconCheck, IconChat, IconClock, IconCpu, IconFiles,
  IconMonitor, IconMoon, IconSettings, IconSun, IconTerminal, IconUraShreeLogo,
} from './lib/icons'
import type {
  AppSettings, Attachment, Block, SnapshotDiff, Status, Theme, ToolRun, TreeNode, Turn,
} from './types'
import './styles.css'

type SidePanel = 'files' | 'timeline' | 'memory'
type DockPanel = 'terminal' | 'diff'

const THEME_KEY = 'shree.theme'
const MAX_ATTACHMENT_BYTES = 400_000

function applyTheme(theme: Theme): void {
  const dark = theme === 'dark' ||
    (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
}

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(file)
  })
}

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? '').split(',')[1] ?? '')
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

const TEXT_PATTERN = /\.(txt|md|py|js|jsx|ts|tsx|json|ya?ml|toml|ini|cfg|css|scss|html?|xml|sh|ps1|bat|sql|rs|go|java|c|h|cpp|hpp|rb|php|swift|kt|env|gitignore|dockerfile|lock)$/i

interface Toast { id: number; message: string; tone: 'ok' | 'danger' | 'info' }

export default function App() {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem(THEME_KEY) as Theme) || 'system',
  )
  const [status, setStatus] = useState<Status | null>(null)
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [tree, setTree] = useState<TreeNode | null>(null)
  const [workspace, setWorkspace] = useState<string | null>(null)
  const [editorPath, setEditorPath] = useState<string | null>(null)
  const [skillsOpen, setSkillsOpen] = useState(false)

  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const [connected, setConnected] = useState(false)
  const [attachments, setAttachments] = useState<Attachment[]>([])

  const [sidePanel, setSidePanel] = useState<SidePanel>('files')
  const [dockPanel, setDockPanel] = useState<DockPanel>('terminal')
  const [dockOpen, setDockOpen] = useState(true)
  const [diff, setDiff] = useState<SnapshotDiff | null>(null)
  // #settings deep-links straight to the dialog, which is handy for a
  // bookmark and for pointing someone at where the key goes.
  const [settingsOpen, setSettingsOpen] = useState(
    () => window.location.hash === '#settings',
  )
  const [timelineKey, setTimelineKey] = useState(0)
  const [toasts, setToasts] = useState<Toast[]>([])

  const socketRef = useRef<Socket | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const folderInput = useRef<HTMLInputElement>(null)
  const turnCounter = useRef(0)

  const notify = useCallback((message: string, tone: Toast['tone'] = 'info') => {
    const id = Date.now() + Math.random()
    setToasts((prev) => [...prev, { id, message, tone }])
    window.setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 5200)
  }, [])

  /* ── theme ─────────────────────────────────────────────────────────────── */

  useEffect(() => {
    applyTheme(theme)
    localStorage.setItem(THEME_KEY, theme)
    if (theme !== 'system') return
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => applyTheme('system')
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [theme])

  /* ── data ──────────────────────────────────────────────────────────────── */

  const refreshTree = useCallback(() => {
    api.tree().then((data) => setTree(data.tree)).catch(() => undefined)
  }, [])

  useEffect(() => {
    api.settings().then(setSettings).catch((err) => notify(String(err.message), 'danger'))
    api.currentWorkspace().then((data) => setWorkspace(data.workspace)).catch(() => undefined)
    refreshTree()
    const poll = () => api.status().then(setStatus).catch(() => undefined)
    void poll()
    const timer = window.setInterval(poll, 5000)
    return () => window.clearInterval(timer)
  }, [notify, refreshTree])

  /* ── agent socket ──────────────────────────────────────────────────────── */

  const appendToLastTurn = useCallback((mutate: (turn: Turn) => Turn) => {
    setTurns((prev) => {
      if (prev.length === 0) return prev
      const next = [...prev]
      next[next.length - 1] = mutate(next[next.length - 1])
      return next
    })
  }, [])

  useEffect(() => {
    const socket = new Socket('/ws/agent?session=default')
    socketRef.current = socket

    const offStatus = socket.onStatus(setConnected)
    const offMessage = socket.onMessage((message) => {
      const type = message.type as string

      if (type === 'run_start') {
        appendToLastTurn((turn) => ({
          ...turn,
          meta: { ...turn.meta, provider: String(message.provider), model: String(message.model) },
        }))
        return
      }

      if (type === 'text' || type === 'thinking') {
        const kind: Block['kind'] = type === 'text' ? 'text' : 'thinking'
        appendToLastTurn((turn) => {
          const blocks = [...turn.blocks]
          const last = blocks[blocks.length - 1]
          // Append into the open block of the same kind so streamed tokens do
          // not each become their own paragraph.
          if (last && last.kind === kind) {
            blocks[blocks.length - 1] = { ...last, text: last.text + String(message.text) }
          } else {
            blocks.push({ kind, text: String(message.text) })
          }
          return { ...turn, blocks }
        })
        return
      }

      if (type === 'tool_start') {
        const tool: ToolRun = {
          id: String(message.id),
          name: String(message.name),
          arguments: (message.arguments as Record<string, unknown>) ?? {},
          mutating: Boolean(message.mutating),
          status: 'running',
          text: '',
          data: {},
        }
        appendToLastTurn((turn) => ({ ...turn, blocks: [...turn.blocks, { kind: 'tool', text: '', tool }] }))
        return
      }

      if (type === 'tool_approval_prompt') {
        const toolId = String(message.id)
        appendToLastTurn((turn) => ({
          ...turn,
          blocks: turn.blocks.map((block) =>
            block.kind === 'tool' && block.tool?.id === toolId
              ? {
                ...block,
                tool: {
                  ...block.tool!,
                  status: 'awaiting_approval',
                  needs_approval: true,
                },
              }
              : block,
          ),
        }))
        notify('Permission requested: Shree wants to write/edit a file', 'info')
        return
      }

      if (type === 'tool_end') {
        appendToLastTurn((turn) => ({
          ...turn,
          blocks: turn.blocks.map((block) =>
            block.kind === 'tool' && block.tool?.id === message.id
              ? {
                ...block,
                tool: {
                  ...block.tool!,
                  status: message.ok ? 'ok' : 'failed',
                  needs_approval: false,
                  text: String(message.text ?? ''),
                  data: (message.data as Record<string, unknown>) ?? {},
                },
              }
              : block,
          ),
        }))
        if (['write_file', 'edit_file', 'run_command'].includes(String(message.name))) {
          refreshTree()
          setTimelineKey((k) => k + 1)
        }
        return
      }

      if (type === 'error') {
        appendToLastTurn((turn) => ({
          ...turn,
          streaming: false,
          blocks: [...turn.blocks, { kind: 'error', text: String(message.message) }],
        }))
        setBusy(false)
        return
      }

      if (type === 'done' || type === 'cancelled' || type === 'stopped') {
        appendToLastTurn((turn) => ({
          ...turn,
          streaming: false,
          meta: {
            ...turn.meta,
            durationMs: Number(message.duration_ms ?? 0) || turn.meta?.durationMs,
            usage: (message.usage as Record<string, number>) ?? turn.meta?.usage,
          },
        }))
        setBusy(false)
        refreshTree()
        setTimelineKey((k) => k + 1)
      }
    })

    return () => { offStatus(); offMessage(); socket.close() }
  }, [appendToLastTurn, notify, refreshTree])

  /* ── actions ───────────────────────────────────────────────────────────── */

  const send = (text: string) => {
    const id = `t${turnCounter.current++}`
    setTurns((prev) => [
      ...prev,
      {
        id: `${id}-u`,
        role: 'user',
        text,
        blocks: [],
        attachments: attachments.map((a) => ({ name: a.path, size: a.size })),
      },
      { id: `${id}-a`, role: 'assistant', text: '', blocks: [], streaming: true },
    ])
    setBusy(true)
    socketRef.current?.send({
      action: 'chat',
      message: text,
      attachments: attachments
        .filter((a) => a.isText)
        .map((a) => ({ path: a.path, content: a.content })),
    })
    setAttachments([])
  }

  const stop = () => {
    socketRef.current?.send({ action: 'stop' })
    setBusy(false)
  }

  const newChat = () => {
    socketRef.current?.send({ action: 'reset' })
    setTurns([])
    setAttachments([])
    setBusy(false)
  }

  const ingest = async (files: FileList | null, asFolder: boolean) => {
    if (!files?.length) return
    const list = Array.from(files)

    if (asFolder) {
      // A folder goes into the workspace so the agent's tools can reach it,
      // rather than being pasted into the prompt.
      const encoded = await Promise.all(
        list.slice(0, 400).map(async (file) => ({
          path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
          content_base64: await readFileAsBase64(file),
        })),
      )
      try {
        const result = await api.upload('uploads', encoded)
        notify(
          `Added ${result.count} file(s) under uploads` +
          (result.rejected.length ? `; ${result.rejected.length} rejected.` : '.'),
          result.count ? 'ok' : 'danger',
        )
        result.rejected.slice(0, 3).forEach((r) => notify(`${r.path}: ${r.reason}`, 'danger'))
        refreshTree()
        setTimelineKey((k) => k + 1)
      } catch (err) {
        notify((err as Error).message, 'danger')
      }
      return
    }

    const attached: Attachment[] = []
    for (const file of list.slice(0, 20)) {
      const isText = TEXT_PATTERN.test(file.name) && file.size <= MAX_ATTACHMENT_BYTES
      attached.push({
        name: file.name,
        path: file.name,
        size: file.size,
        isText,
        content: isText ? await readFileAsText(file) : '',
        base64: isText ? '' : await readFileAsBase64(file),
      })
      if (!isText) {
        notify(`${file.name} is not readable text; only its name is sent.`, 'info')
      }
    }
    setAttachments((prev) => [...prev, ...attached])
  }

  const compare = async (fromId: string, toId: string) => {
    try {
      setDiff(await api.diff(fromId, toId))
      setDockPanel('diff')
      setDockOpen(true)
    } catch (err) {
      notify((err as Error).message, 'danger')
    }
  }

  const openFile = async (path: string) => {
    try {
      const file = await api.readFile(path)
      const language = path.split('.').pop() ?? ''
      const id = `t${turnCounter.current++}`
      setTurns((prev) => [
        ...prev,
        {
          id: `${id}-a`,
          role: 'assistant',
          text: '',
          blocks: [{
            kind: 'text',
            text: `\`${file.path}\` — ${file.total_lines} lines\n\n\`\`\`${language}\n${file.content}\n\`\`\``,
          }],
        },
      ])
    } catch (err) {
      notify((err as Error).message, 'danger')
    }
  }

  const handleSelectWorkspace = async (path: string) => {
    try {
      const res = await api.selectWorkspace(path)
      setWorkspace(res.workspace)
      refreshTree()
      notify(res.workspace ? `Workspace set to: ${res.workspace}` : 'Switched to General Chat Mode (no folder)', 'ok')
    } catch (err) {
      notify((err as Error).message, 'danger')
    }
  }

  const handleDeleteFile = async (path: string) => {
    try {
      const res = await api.deleteFile(path)
      notify(`Deleted ${res.deleted}`, 'ok')
      refreshTree()
      setTimelineKey((k) => k + 1)
    } catch (err) {
      notify((err as Error).message, 'danger')
    }
  }

  const handleApproveTool = (id: string, approved: boolean) => {
    socketRef.current?.send({ action: 'tool_approval', id, approved })
    appendToLastTurn((turn) => ({
      ...turn,
      blocks: turn.blocks.map((block) =>
        block.kind === 'tool' && block.tool?.id === id
          ? {
            ...block,
            tool: {
              ...block.tool!,
              status: approved ? 'running' : 'failed',
              needs_approval: false,
              text: approved ? 'Approved by user. Executing...' : 'Rejected by user.',
            },
          }
          : block,
      ),
    }))
    notify(approved ? 'Tool execution approved' : 'Tool execution rejected', approved ? 'ok' : 'info')
  }

  /* ── header state ──────────────────────────────────────────────────────── */

  const modelLabel = useMemo(() => {
    if (!settings) return 'the model'
    const { provider, model } = settings.active
    if (provider === 'local') return 'Ura-Shree (local)'
    return model || 'no model selected'
  }, [settings])

  const vram = status?.hardware
  const themeIcon = theme === 'dark' ? <IconMoon size={15} />
    : theme === 'light' ? <IconSun size={15} /> : <IconMonitor size={15} />

  return (
    <div className="app">
      <nav className="rail">
        <div className="rail-brand" title="Ura-Shree">
          <IconUraShreeLogo size={24} />
        </div>
        {([
          ['files', <IconFiles key="f" size={18} />, 'Files'],
          ['timeline', <IconClock key="t" size={18} />, 'Time machine'],
          ['memory', <IconBrain key="m" size={18} />, 'Project memory'],
        ] as const).map(([id, icon, label]) => (
          <button
            key={id}
            className={`rail-btn${sidePanel === id ? ' active' : ''}`}
            onClick={() => setSidePanel(id)}
            title={label}
            aria-label={label}
          >
            {icon}
          </button>
        ))}
        <button
          className={`rail-btn${skillsOpen ? ' active' : ''}`}
          onClick={() => setSkillsOpen(true)}
          title="Skills & Specialist Workflows"
          aria-label="Agent Skills"
        >
          <IconBolt size={18} />
        </button>
        <span className="spacer" />
        <button
          className={`rail-btn${dockOpen ? ' active' : ''}`}
          onClick={() => setDockOpen(!dockOpen)}
          title="Terminal"
          aria-label="Toggle terminal"
        >
          <IconTerminal size={18} />
        </button>
        <button
          className="rail-btn"
          onClick={() => setTheme(theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light')}
          title={`Theme: ${theme}`}
          aria-label={`Switch theme, currently ${theme}`}
        >
          {themeIcon}
        </button>
        <button
          className="rail-btn"
          onClick={() => setSettingsOpen(true)}
          title="Settings"
          aria-label="Open settings"
        >
          <IconSettings size={18} />
        </button>
      </nav>

      <div className="main">
        <header className="header">
          <button className="model-pick" onClick={() => setSettingsOpen(true)}>
            <span className={`dot ${connected ? 'dot-ok' : 'dot-danger'}`} />
            <strong className="truncate">{modelLabel === 'shree:latest' ? 'Shree' : modelLabel}</strong>
            {settings && settings.active.provider !== 'local' && (
              <span className="vendor">
                {settings.active.model.startsWith('shree') ? 'DIU' : settings.active.provider}
              </span>
            )}
          </button>

          <button className="btn btn-ghost btn-sm" onClick={newChat}>New chat</button>

          <span className="spacer" />

          {status && (
            <>
              {settings?.active.use_tools === false && <span className="chip chip-warn">tools off</span>}
              {settings?.active.auto_approve === false && <span className="chip chip-warn">approval required</span>}
              <span className="chip" title="Compute device">
                <IconCpu size={11} />
                {status.runtime.device_name?.replace(/NVIDIA GeForce /, '') ?? status.runtime.device}
              </span>
              {vram && vram.cuda_available && (
                <span className="chip" title="GPU memory in use">
                  {(vram.allocated_mb / 1024).toFixed(1)}/{(vram.total_mb / 1024).toFixed(0)} GB
                </span>
              )}
              <span className="chip" title="Process memory">
                RAM {(vram!.process_rss_mb / 1024).toFixed(1)} GB
              </span>
              <span className="chip" title="Snapshot store">
                <IconClock size={11} />
                {formatBytes(status.time_machine.store_bytes)}
              </span>
            </>
          )}
        </header>

        <div className="workbench">
          <aside className="pane pane-side">
            {sidePanel === 'files' && (
              <FilesPanel
                tree={tree}
                workspace={workspace}
                onOpenFile={openFile}
                onEditFile={(path) => setEditorPath(path)}
                onDeleteFile={handleDeleteFile}
                onSelectWorkspace={handleSelectWorkspace}
                onRefresh={refreshTree}
                onAddFiles={() => fileInput.current?.click()}
                onAddFolder={() => folderInput.current?.click()}
                onClose={() => setSidePanel('' as SidePanel)}
              />
            )}
            {sidePanel === 'timeline' && (
              <TimelinePanel onCompare={compare} onNotify={notify} refreshKey={timelineKey} />
            )}
            {sidePanel === 'memory' && <MemoryPanel refreshKey={timelineKey} />}
          </aside>

          <main className="pane pane-main">
            <Chat
              turns={turns}
              busy={busy}
              connected={connected}
              modelLabel={modelLabel}
              attachments={attachments}
              onSend={send}
              onStop={stop}
              onApproveTool={handleApproveTool}
              onOpenSkills={() => setSkillsOpen(true)}
              onAttach={() => fileInput.current?.click()}
              onAttachFolder={() => folderInput.current?.click()}
              onRemoveAttachment={(name) =>
                setAttachments((prev) => prev.filter((a) => a.name !== name))}
            />
          </main>

          {dockOpen && (
            <aside className="pane pane-dock">
              <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
                {diff && (
                  <div className="tabs" style={{ padding: '6px 8px 0' }}>
                    <button
                      className={`tab${dockPanel === 'terminal' ? ' active' : ''}`}
                      onClick={() => setDockPanel('terminal')}
                    >
                      Terminal
                    </button>
                    <button
                      className={`tab${dockPanel === 'diff' ? ' active' : ''}`}
                      onClick={() => setDockPanel('diff')}
                    >
                      Diff
                    </button>
                  </div>
                )}
                {dockPanel === 'diff' && diff ? (
                  <DiffView diff={diff} onClose={() => { setDiff(null); setDockPanel('terminal') }} />
                ) : (
                  <TerminalPanel
                    workspace={workspace}
                    onClose={() => setDockOpen(false)}
                  />
                )}
              </div>
            </aside>
          )}
        </div>
      </div>

      <input
        ref={fileInput}
        type="file"
        multiple
        hidden
        onChange={(e) => { void ingest(e.target.files, false); e.target.value = '' }}
      />
      <input
        ref={folderInput}
        type="file"
        hidden
        {...{ webkitdirectory: '', directory: '' }}
        onChange={(e) => { void ingest(e.target.files, true); e.target.value = '' }}
      />

      {editorPath && (
        <CodeEditorModal
          path={editorPath}
          onClose={() => setEditorPath(null)}
          onSaved={() => {
            refreshTree()
            setTimelineKey((k) => k + 1)
          }}
          onNotify={notify}
        />
      )}

      {skillsOpen && (
        <SkillsModal
          onClose={() => setSkillsOpen(false)}
          onNotify={notify}
        />
      )}

      {settingsOpen && settings && (
        <SettingsDialog
          settings={settings}
          status={status}
          theme={theme}
          onThemeChange={setTheme}
          onSettingsChange={setSettings}
          onNotify={notify}
          onClose={() => setSettingsOpen(false)}
        />
      )}

      <div className="toast-stack">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.tone}`}>
            {toast.tone === 'danger' ? <IconAlert size={14} style={{ flex: 'none', marginTop: 1, color: 'var(--danger)' }} />
              : toast.tone === 'ok' ? <IconCheck size={14} style={{ flex: 'none', marginTop: 1, color: 'var(--ok)' }} />
                : <IconChat size={14} style={{ flex: 'none', marginTop: 1, color: 'var(--text-3)' }} />}
            <span>{toast.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
