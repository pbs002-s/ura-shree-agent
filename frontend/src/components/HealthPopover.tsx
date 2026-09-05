import { useEffect, useRef } from 'react'
import { formatBytes, formatNumber } from '../lib/api'
import { IconCpu } from '../lib/icons'
import type { AppSettings, Status } from '../types'

interface HealthProps {
  status: Status
  settings: AppSettings | null
  onOpenSettings: () => void
  onClose: () => void
}

function Meter({ label, used, total, unit }: { label: string; used: number; total: number; unit: string }) {
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0
  return (
    <div className="meter">
      <div className="meter-row">
        <span>{label}</span>
        <span className="spacer" />
        <span className="mono">
          {used.toFixed(1)} / {total.toFixed(1)} {unit}
        </span>
      </div>
      <div className="meter-track">
        <div
          className={`meter-fill${pct > 88 ? ' hot' : ''}`}
          style={{ transform: `scaleX(${pct / 100})` }}
        />
      </div>
    </div>
  )
}

/**
 * Health readout behind the header chips.
 *
 * Everything here already arrives on the five-second /api/status poll, so the
 * popover is a view onto state the app holds rather than another request.
 */
export function HealthPopover({ status, settings, onOpenSettings, onClose }: HealthProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onDown = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) onClose()
    }
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    // Deferred so the click that opened the popover does not close it again.
    const timer = window.setTimeout(() => document.addEventListener('mousedown', onDown))
    document.addEventListener('keydown', onKey)
    return () => {
      window.clearTimeout(timer)
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  const hw = status.hardware
  const runtime = status.runtime
  const active = settings?.active

  return (
    <div className="popover" ref={ref} role="dialog" aria-label="Hardware and health">
      <div className="popover-head">
        <IconCpu size={14} />
        <strong>{runtime.device_name ?? runtime.device ?? 'compute'}</strong>
        <span className="spacer" />
        <span className={`chip ${hw.cuda_available ? 'chip-ok' : ''}`}>
          {hw.cuda_available ? 'CUDA' : 'CPU'}
        </span>
      </div>

      <div className="popover-body">
        {hw.cuda_available && (
          <Meter label="VRAM allocated" used={hw.allocated_mb / 1024} total={hw.total_mb / 1024} unit="GB" />
        )}
        {hw.cuda_available && hw.reserved_mb > 0 && (
          <Meter label="VRAM reserved" used={hw.reserved_mb / 1024} total={hw.total_mb / 1024} unit="GB" />
        )}
        <Meter
          label="System RAM"
          used={hw.system_used_mb / 1024}
          total={hw.system_total_mb / 1024}
          unit="GB"
        />

        <dl className="kv" style={{ marginTop: 12 }}>
          <dt>Process</dt><dd>{(hw.process_rss_mb / 1024).toFixed(2)} GB resident</dd>
          <dt>Precision</dt><dd>{runtime.dtype ?? 'unknown'}</dd>
          <dt>CPU</dt>
          <dd>{runtime.cpu_threads ?? '?'} threads of {runtime.logical_cores ?? '?'} logical</dd>
          <dt>Provider</dt><dd>{active?.provider ?? 'unset'}</dd>
          <dt>Model</dt><dd className="truncate">{active?.model || 'local checkpoint'}</dd>
          {status.local_model.loaded && status.local_model.parameters != null && (
            <>
              <dt>Parameters</dt><dd>{formatNumber(status.local_model.parameters)}</dd>
            </>
          )}
          <dt>Snapshots</dt><dd>{formatBytes(status.time_machine.store_bytes)} stored</dd>
        </dl>

        {!!runtime.notes?.length && (
          <div className="popover-notes">
            {runtime.notes.slice(0, 4).map((note) => <div key={note}>· {note}</div>)}
          </div>
        )}
      </div>

      <div className="popover-foot">
        <button className="btn btn-ghost btn-sm" onClick={() => { onClose(); onOpenSettings() }}>
          Open settings
        </button>
        <span className="spacer" />
        <span className="faint mono">v{status.version}</span>
      </div>
    </div>
  )
}
