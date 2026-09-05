import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { IconSearch } from '../lib/icons'

export interface Command {
  id: string
  label: string
  group: string
  icon?: ReactNode
  hint?: string
  /** Extra words that should match the filter but are not shown. */
  keywords?: string
  run: () => void
}

interface PaletteProps {
  commands: Command[]
  onClose: () => void
}

/**
 * Ctrl/Cmd+K palette.
 *
 * Filtering is a plain substring test rather than fuzzy matching: the command
 * list is a few dozen entries, and a substring is what people actually type.
 */
export function CommandPalette({ commands, onClose }: PaletteProps) {
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return commands
    return commands.filter((command) =>
      `${command.label} ${command.group} ${command.keywords ?? ''}`.toLowerCase().includes(needle))
  }, [commands, query])

  // A shrinking result list must never leave the cursor pointing past the end.
  const active = Math.min(cursor, Math.max(0, matches.length - 1))

  useEffect(() => {
    listRef.current
      ?.querySelector<HTMLElement>('.palette-item.active')
      ?.scrollIntoView({ block: 'nearest' })
  }, [active, matches.length])

  const runAt = (index: number) => {
    const command = matches[index]
    if (!command) return
    onClose()
    command.run()
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor((c) => Math.min(c + 1, matches.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor((c) => Math.max(c - 1, 0))
    } else if (event.key === 'Enter') {
      event.preventDefault()
      runAt(active)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      onClose()
    }
  }

  let lastGroup = ''

  return (
    <div className="palette-backdrop" onMouseDown={onClose}>
      <div
        className="palette"
        role="dialog"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="palette-search">
          <IconSearch size={15} />
          <input
            autoFocus
            value={query}
            placeholder="Search commands, panels, themes…"
            onChange={(e) => { setQuery(e.target.value); setCursor(0) }}
            onKeyDown={onKeyDown}
            aria-label="Command"
          />
          <kbd className="kbd">esc</kbd>
        </div>

        <div className="palette-list" ref={listRef}>
          {matches.length === 0 && <div className="empty">No command matches “{query}”.</div>}
          {matches.map((command, index) => {
            const header = command.group !== lastGroup ? command.group : ''
            lastGroup = command.group
            return (
              <div key={command.id}>
                {header && <div className="palette-group">{header}</div>}
                <button
                  className={`palette-item${index === active ? ' active' : ''}`}
                  onMouseEnter={() => setCursor(index)}
                  onClick={() => runAt(index)}
                >
                  <span className="palette-icon">{command.icon}</span>
                  <span className="truncate">{command.label}</span>
                  <span className="spacer" />
                  {command.hint && <span className="palette-hint">{command.hint}</span>}
                </button>
              </div>
            )
          })}
        </div>

        <div className="palette-foot">
          <span><kbd className="kbd">↑</kbd><kbd className="kbd">↓</kbd> navigate</span>
          <span><kbd className="kbd">↵</kbd> run</span>
          <span className="spacer" />
          <span>{matches.length} of {commands.length}</span>
        </div>
      </div>
    </div>
  )
}
