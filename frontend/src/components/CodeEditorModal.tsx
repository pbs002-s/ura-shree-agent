import { useEffect, useState } from 'react'
import { api, formatBytes } from '../lib/api'
import { IconCheck, IconClose } from '../lib/icons'

interface CodeEditorModalProps {
  path: string | null
  onClose: () => void
  onSaved?: () => void
  onNotify?: (message: string, tone?: 'ok' | 'danger' | 'info') => void
}

export function CodeEditorModal({ path, onClose, onSaved, onNotify }: CodeEditorModalProps) {
  const [content, setContent] = useState('')
  const [initialContent, setInitialContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!path) return
    let active = true
    setLoading(true)
    setError('')

    api.readFile(path)
      .then((res) => {
        if (!active) return
        setContent(res.content)
        setInitialContent(res.content)
      })
      .catch((err) => {
        if (!active) return
        setError(err.message || 'Failed to read file')
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => { active = false }
  }, [path])

  if (!path) return null

  const isDirty = content !== initialContent
  const linesCount = content.split('\n').length
  const language = path.split('.').pop() || 'text'

  const handleSave = async () => {
    if (!path || saving) return
    setSaving(true)
    setError('')
    try {
      await api.writeFile(path, content)
      setInitialContent(content)
      onNotify?.(`Saved ${path}`, 'ok')
      onSaved?.()
    } catch (err) {
      const msg = (err as Error).message || 'Failed to save file'
      setError(msg)
      onNotify?.(msg, 'danger')
    } finally {
      setSaving(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault()
      handleSave()
    } else if (e.key === 'Tab') {
      e.preventDefault()
      const target = e.currentTarget
      const start = target.selectionStart
      const end = target.selectionEnd
      const val = target.value
      target.value = val.substring(0, start) + '  ' + val.substring(end)
      target.selectionStart = target.selectionEnd = start + 2
      setContent(target.value)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card code-editor-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <span className="code-editor-pill">{language}</span>
            <span className="code-editor-path truncate">{path}</span>
            {isDirty ? (
              <span className="chip chip-warn">Unsaved changes</span>
            ) : (
              <span className="chip chip-ok"><IconCheck size={11} /> Saved</span>
            )}
          </div>
          <div className="modal-actions">
            <button
              className="btn btn-primary btn-sm"
              disabled={!isDirty || saving || loading}
              onClick={handleSave}
            >
              {saving ? 'Saving…' : 'Save Changes (Ctrl+S)'}
            </button>
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose} title="Close editor">
              <IconClose size={15} />
            </button>
          </div>
        </div>

        {error && <div className="banner banner-danger">{error}</div>}

        <div className="code-editor-body">
          {loading ? (
            <div className="code-editor-loading faint">Loading file…</div>
          ) : (
            <div className="code-editor-wrapper">
              <div className="code-editor-line-numbers">
                {Array.from({ length: linesCount }, (_, i) => (
                  <div key={i + 1}>{i + 1}</div>
                ))}
              </div>
              <textarea
                className="code-editor-textarea"
                value={content}
                onChange={(e) => setContent(e.target.value)}
                onKeyDown={handleKeyDown}
                spellCheck={false}
                autoCapitalize="off"
                autoComplete="off"
              />
            </div>
          )}
        </div>

        <div className="code-editor-footer">
          <span className="faint">{linesCount} lines</span>
          <span className="faint">·</span>
          <span className="faint">{formatBytes(new TextEncoder().encode(content).length)}</span>
          <span className="spacer" />
          <span className="faint">Press Tab to indent, Ctrl+S to save</span>
        </div>
      </div>
    </div>
  )
}
