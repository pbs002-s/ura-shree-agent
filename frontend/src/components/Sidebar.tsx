import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, formatBytes, relativeTime } from '../lib/api'
import {
  IconBranch, IconChevronDown, IconChevronRight, IconClock, IconClose, IconEdit, IconFolder,
  IconFolderPlus, IconPlus, IconRefresh, IconRestore, IconSearch, IconTrash,
} from '../lib/icons'
import type { Snapshot, TreeNode } from '../types'

/* ── Files ──────────────────────────────────────────────────────────────── */

function TreeRow({
  node, depth, expanded, selected, onToggle, onOpen, onEdit, onDelete,
}: {
  node: TreeNode
  depth: number
  expanded: Set<string>
  selected: string
  onToggle: (path: string) => void
  onOpen: (node: TreeNode) => void
  onEdit?: (path: string) => void
  onDelete?: (path: string) => void
}) {
  const isOpen = expanded.has(node.path)
  return (
    <div className={`tree-row-container${selected === node.path ? ' selected' : ''}`}>
      <button
        className="tree-row"
        style={{ paddingLeft: 8 + depth * 12 }}
        onClick={() => (node.isDir ? onToggle(node.path) : onOpen(node))}
        title={node.path}
      >
        <span className="tree-caret">
          {node.isDir && (isOpen ? <IconChevronDown size={11} /> : <IconChevronRight size={11} />)}
        </span>
        <span className="truncate">{node.name}</span>
        {!node.isDir && node.size != null && (
          <span className="tree-size">{formatBytes(node.size)}</span>
        )}
      </button>
      {!node.isDir && (
        <div className="tree-row-actions">
          {onEdit && (
            <button
              className="tree-action-btn"
              onClick={(e) => {
                e.stopPropagation()
                onEdit(node.path)
              }}
              title="Edit code"
            >
              <IconEdit size={12} />
            </button>
          )}
          {onDelete && (
            <button
              className="tree-action-btn danger"
              onClick={(e) => {
                e.stopPropagation()
                if (window.confirm(`Delete file "${node.name}"?`)) {
                  onDelete(node.path)
                }
              }}
              title="Delete file"
            >
              <IconTrash size={12} />
            </button>
          )}
        </div>
      )}
      {node.isDir && isOpen && node.children?.map((child) => (
        <TreeRow
          key={child.path}
          node={child}
          depth={depth + 1}
          expanded={expanded}
          selected={selected}
          onToggle={onToggle}
          onOpen={onOpen}
          onEdit={onEdit}
          onDelete={onDelete}
        />
      ))}
    </div>
  )
}

function flatten(node: TreeNode, out: TreeNode[] = []): TreeNode[] {
  if (!node.isDir) out.push(node)
  node.children?.forEach((child) => flatten(child, out))
  return out
}

export function FilesPanel({
  tree, workspace, onOpenFile, onEditFile, onDeleteFile, onSelectWorkspace, onRefresh, onAddFiles, onAddFolder, onClose,
}: {
  tree: TreeNode | null
  workspace?: string | null
  onOpenFile: (path: string) => void
  onEditFile?: (path: string) => void
  onDeleteFile?: (path: string) => void
  onSelectWorkspace?: (path: string) => void
  onRefresh: () => void
  onAddFiles: () => void
  onAddFolder: () => void
  onClose?: () => void
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['']))
  const [selected, setSelected] = useState('')
  const [query, setQuery] = useState('')

  const matches = useMemo(() => {
    if (!tree || !query.trim()) return null
    const needle = query.toLowerCase()
    return flatten(tree).filter((n) => n.path.toLowerCase().includes(needle)).slice(0, 200)
  }, [tree, query])

  const toggle = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const open = (node: TreeNode) => {
    setSelected(node.path)
    if (onEditFile) {
      onEditFile(node.path)
    } else {
      onOpenFile(node.path)
    }
  }

  const [openingFolder, setOpeningFolder] = useState(false)

  const openFolderPicker = async () => {
    if (openingFolder) return
    setOpeningFolder(true)
    try {
      const res = await api.browseWorkspace()
      if (res.ok && res.workspace && onSelectWorkspace) {
        onSelectWorkspace(res.workspace)
      }
    } catch (err) {
      console.error('Failed to open folder picker:', err)
    } finally {
      setOpeningFolder(false)
    }
  }

  const workspaceName = workspace ? (workspace.split(/[/\\]/).filter(Boolean).pop() || workspace) : null

  return (
    <>
      <div className="pane-head">
        <span className="section-label">Files</span>
        <span className="spacer" />
        <button
          className="btn btn-ghost btn-sm"
          onClick={openFolderPicker}
          disabled={openingFolder}
          title={workspace ? `Active: ${workspace}. Click to switch folder.` : 'Open project folder in File Explorer'}
          style={{ gap: 4, fontSize: 'var(--fs-xs)', padding: '2px 7px' }}
        >
          <IconFolder size={12} />
          <span>{openingFolder ? 'Opening…' : (workspaceName ? 'Switch Folder' : 'Open Folder')}</span>
        </button>
        {workspace && (
          <>
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onAddFiles} title="Upload files">
              <IconPlus size={13} />
            </button>
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onAddFolder} title="Add a folder">
              <IconFolderPlus size={13} />
            </button>
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onRefresh} title="Refresh">
              <IconRefresh size={13} />
            </button>
          </>
        )}
        {onClose && (
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose} title="Hide files panel" aria-label="Hide files panel">
            <IconClose size={13} />
          </button>
        )}
      </div>

      {workspace && (
        <div className="workspace-badge-bar">
          <IconFolder size={12} className="accent" />
          <span className="workspace-path truncate" title={workspace}>{workspace}</span>
          <button
            className="btn btn-ghost btn-sm faint"
            onClick={() => onSelectWorkspace?.('')}
            title="Detach folder to enter pure General Chat Mode"
            style={{ fontSize: 10, padding: '1px 5px', height: 20 }}
          >
            Detach
          </button>
        </div>
      )}

      {workspace && (
        <div style={{ padding: '7px 8px', borderBottom: '1px solid var(--line)' }}>
          <div style={{ position: 'relative' }}>
            <IconSearch
              size={13}
              style={{ position: 'absolute', left: 8, top: 8, color: 'var(--text-3)' }}
            />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by path"
              style={{ paddingLeft: 26, height: 28 }}
            />
          </div>
        </div>
      )}

      <div className="pane-body">
        {!workspace ? (
          <div className="empty" style={{ padding: '36px 16px', textAlign: 'center' }}>
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 12 }}>
              <IconFolder size={36} style={{ color: 'var(--accent)' }} />
            </div>
            <div style={{ fontWeight: 600, color: 'var(--text-1)', marginBottom: 8, fontSize: 'var(--fs-sm)' }}>
              General Chat Mode
            </div>
            <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-3)', lineHeight: 1.6, marginBottom: 16 }}>
              No project folder is open. Normal chat operates freely with Shree. To work on code files, run tests, or edit a repository, open a folder.
            </div>
            <button
              className="btn btn-primary btn-sm"
              onClick={openFolderPicker}
              disabled={openingFolder}
              style={{ margin: '0 auto', display: 'inline-flex' }}
            >
              <IconFolder size={13} />
              <span>{openingFolder ? 'Opening File Explorer…' : 'Open Project Folder'}</span>
            </button>
          </div>
        ) : !tree ? (
          <div className="empty">Loading the workspace…</div>
        ) : matches ? (
          <div className="tree">
            {matches.length === 0 && <div className="empty">Nothing matches “{query}”.</div>}
            {matches.map((node) => (
              <div key={node.path} className={`tree-row-container${selected === node.path ? ' selected' : ''}`}>
                <button
                  className="tree-row"
                  style={{ paddingLeft: 8 }}
                  onClick={() => open(node)}
                  title={node.path}
                >
                  <span className="tree-caret" />
                  <span className="truncate">{node.path}</span>
                </button>
                <div className="tree-row-actions">
                  <button
                    className="tree-action-btn"
                    onClick={() => onEditFile?.(node.path)}
                    title="Edit code"
                  >
                    <IconEdit size={12} />
                  </button>
                  {onDeleteFile && (
                    <button
                      className="tree-action-btn danger"
                      onClick={() => {
                        if (window.confirm(`Delete file "${node.name}"?`)) onDeleteFile(node.path)
                      }}
                      title="Delete file"
                    >
                      <IconTrash size={12} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="tree">
            {(!tree.children || tree.children.length === 0) && (
              <div className="empty" style={{ padding: '32px 14px', textAlign: 'center' }}>
                <div style={{ fontWeight: 600, color: 'var(--text-1)', marginBottom: 6 }}>Workspace is empty</div>
                <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-3)', lineHeight: 1.5 }}>
                  Drop files here, upload via the + button, or ask Shree to generate project files.
                </div>
              </div>
            )}
            {tree.children?.map((child) => (
              <TreeRow
                key={child.path}
                node={child}
                depth={0}
                expanded={expanded}
                selected={selected}
                onToggle={toggle}
                onOpen={open}
                onEdit={onEditFile}
                onDelete={onDeleteFile}
              />
            ))}
          </div>
        )}
      </div>
    </>
  )
}

/* ── Time machine ───────────────────────────────────────────────────────── */

export function TimelinePanel({
  onCompare, onNotify, refreshKey,
}: {
  onCompare: (fromId: string, toId: string) => void
  onNotify: (message: string, tone?: 'ok' | 'danger') => void
  refreshKey: number
}) {
  const [nodes, setNodes] = useState<Snapshot[]>([])
  const [head, setHead] = useState<string | null>(null)
  const [storeBytes, setStoreBytes] = useState(0)
  const [selected, setSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      const data = await api.timeline()
      // Newest first: the interesting end of a session history is the recent end.
      setNodes([...data.nodes].reverse())
      setHead(data.head)
      setStoreBytes(data.store_bytes)
    } catch (err) {
      onNotify(`Could not load the timeline: ${(err as Error).message}`, 'danger')
    }
  }, [onNotify])

  useEffect(() => { void load() }, [refreshKey, load])

  const pick = (id: string) => {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      const next = [...prev, id].slice(-2)
      if (next.length === 2) {
        // Nodes render newest first, so the later entry is the older snapshot.
        const older = nodes.findIndex((n) => n.id === next[0]) > nodes.findIndex((n) => n.id === next[1])
          ? next[0] : next[1]
        const newer = older === next[0] ? next[1] : next[0]
        onCompare(older, newer)
      }
      return next
    })
  }

  const takeSnapshot = async () => {
    setBusy(true)
    try {
      const snap = await api.snapshot(`Manual · ${new Date().toLocaleTimeString()}`)
      onNotify(snap.unchanged ? 'Nothing changed since the last snapshot.' : `Snapshot ${snap.id.slice(0, 8)} saved.`, 'ok')
      await load()
    } catch (err) {
      onNotify((err as Error).message, 'danger')
    } finally {
      setBusy(false)
    }
  }

  const restore = async (id: string) => {
    const plan = await api.restore(id, true) as unknown as {
      will_write: string[]; will_delete: string[]
    }
    const message =
      `Restore to ${id.slice(0, 8)}?\n\n` +
      `${plan.will_write.length} file(s) rewritten, ${plan.will_delete.length} removed.\n` +
      'The current state is snapshotted first, so this is reversible.'
    if (!window.confirm(message)) return

    setBusy(true)
    try {
      const result = await api.restore(id)
      onNotify(`Restored: ${result.files_written} written, ${result.files_deleted} deleted.`, 'ok')
      await load()
    } catch (err) {
      onNotify((err as Error).message, 'danger')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="pane-head">
        <span className="section-label">Time machine</span>
        <span className="spacer" />
        <span className="faint" style={{ fontSize: 'var(--fs-xs)' }}>{formatBytes(storeBytes)}</span>
        <button className="btn btn-ghost btn-sm btn-icon" onClick={takeSnapshot} disabled={busy} title="Snapshot now">
          <IconPlus size={13} />
        </button>
        <button className="btn btn-ghost btn-sm btn-icon" onClick={() => void load()} title="Refresh">
          <IconRefresh size={13} />
        </button>
      </div>

      <div className="pane-body">
        {nodes.length === 0 ? (
          <div className="empty">
            <IconClock size={22} />
            No snapshots yet. One is taken automatically before every edit.
          </div>
        ) : (
          <>
            <div style={{ padding: '7px 12px', borderBottom: '1px solid var(--line)' }}
                 className="faint">
              Select two points to diff them.
            </div>
            <div className="timeline">
              {nodes.map((node) => (
                <div key={node.id} className={`tl-node${selected.includes(node.id) ? ' selected' : ''}`}>
                  <div className="tl-rail">
                    <span
                      className={`tl-mark${node.id === head ? ' head' : ''}${node.is_branch_point ? ' branch' : ''}`}
                    />
                  </div>
                  <button
                    onClick={() => pick(node.id)}
                    style={{ flex: 1, minWidth: 0, textAlign: 'left', padding: 0 }}
                  >
                    <div className="tl-label truncate">{node.label}</div>
                    <div className="tl-sub row" style={{ gap: 6 }}>
                      <span className="mono" style={{ fontSize: 'var(--fs-xs)' }}>{node.id.slice(0, 8)}</span>
                      <span>·</span>
                      <span>{relativeTime(node.created_at)}</span>
                      <span>·</span>
                      <span>{node.file_count} files</span>
                      {node.is_branch_point && (
                        <span className="chip" style={{ height: 16, padding: '0 5px' }}>
                          <IconBranch size={9} /> fork
                        </span>
                      )}
                    </div>
                  </button>
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => void restore(node.id)}
                    disabled={busy}
                    title="Restore the workspace to this point"
                    style={{ alignSelf: 'center' }}
                  >
                    <IconRestore size={13} />
                  </button>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </>
  )
}

/* ── Memory ─────────────────────────────────────────────────────────────── */

export function MemoryPanel({ refreshKey }: { refreshKey: number }) {
  const [facts, setFacts] = useState<Record<string, string>>({})
  const [summary, setSummary] = useState('')

  useEffect(() => {
    api.memory()
      .then((data) => { setFacts(data.facts); setSummary(data.summary) })
      .catch(() => undefined)
  }, [refreshKey])

  const entries = Object.entries(facts)

  return (
    <>
      <div className="pane-head"><span className="section-label">Project memory</span></div>
      <div className="pane-body" style={{ padding: '10px 12px' }}>
        {summary && (
          <p className="muted" style={{ marginTop: 0, whiteSpace: 'pre-wrap', fontSize: 'var(--fs-sm)' }}>
            {summary}
          </p>
        )}
        {entries.length === 0 ? (
          <div className="empty">
            Nothing remembered yet. Ask Shree to remember a decision and it lands here.
          </div>
        ) : (
          <dl className="kv" style={{ gridTemplateColumns: '1fr' }}>
            {entries.map(([key, value]) => (
              <div key={key} style={{ marginBottom: 10 }}>
                <dt className="mono" style={{ color: 'var(--accent)' }}>{key}</dt>
                <dd style={{ color: 'var(--text-2)' }}>{value}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
    </>
  )
}
