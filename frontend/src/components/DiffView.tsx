import { IconClose } from '../lib/icons'
import type { SnapshotDiff } from '../types'

function DiffLines({ patch }: { patch: string }) {
  return (
    <>
      {patch.split('\n').map((line, index) => {
        if (!line && index) return null
        const className =
          line.startsWith('+++') || line.startsWith('---') ? 'diff-line faint'
            : line.startsWith('@@') ? 'diff-line diff-hunk'
              : line.startsWith('+') ? 'diff-line diff-add'
                : line.startsWith('-') ? 'diff-line diff-del'
                  : 'diff-line'
        return <div key={index} className={className}>{line || ' '}</div>
      })}
    </>
  )
}

/** Side-by-side comparison of any two points in the workspace history. */
export function DiffView({ diff, onClose }: { diff: SnapshotDiff; onClose: () => void }) {
  const { summary } = diff
  return (
    <>
      <div className="pane-head">
        <span className="section-label">Diff</span>
        <span className="mono faint" style={{ fontSize: 'var(--fs-xs)' }}>
          {diff.from.slice(0, 8)} → {diff.to.slice(0, 8)}
        </span>
        <span className="spacer" />
        {summary.added > 0 && <span className="chip chip-ok">+{summary.added} new</span>}
        {summary.modified > 0 && <span className="chip">{summary.modified} changed</span>}
        {summary.removed > 0 && <span className="chip chip-danger">-{summary.removed} gone</span>}
        <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose} aria-label="Close diff">
          <IconClose size={13} />
        </button>
      </div>

      <div className="pane-body">
        {diff.files.length === 0 ? (
          <div className="empty">These two points are identical.</div>
        ) : (
          diff.files.map((file) => (
            <div key={file.path}>
              <div className="diff-file row">
                <span className="truncate mono">{file.path}</span>
                <span className="spacer" />
                {file.binary ? (
                  <span className="chip">binary</span>
                ) : (
                  <>
                    <span style={{ color: 'var(--add-text)' }}>+{file.additions}</span>
                    <span style={{ color: 'var(--del-text)' }}>-{file.deletions}</span>
                  </>
                )}
              </div>
              {!file.binary && <DiffLines patch={file.diff} />}
            </div>
          ))
        )}
        {summary.truncated && (
          <div className="empty">More files changed than are shown here.</div>
        )}
      </div>
    </>
  )
}
