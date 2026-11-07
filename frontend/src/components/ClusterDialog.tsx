import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { api } from '../lib/api'
import { IconClose, IconCluster, IconDownload, IconExternal, IconRefresh, IconStop } from '../lib/icons'
import type { ClusterJob, ClusterStatus, FineTuneRequest } from '../types'

type Props = {
  onClose: () => void
  onOpenSettings: () => void
  onNotify: (message: string, tone?: 'ok' | 'danger' | 'info') => void
}

const ACTIVE = new Set(['submitting', 'gang_wait', 'idle', 'running', 'held', 'suspended', 'transferring'])
const LOG_POLL_MS = 3000
const MAX_LOG_CHARS = 200_000

const DEFAULT_FORM: FineTuneRequest = {
  name: 'shree-finetune', config: 'small', base: 'best.pt', steps: 600, lr: 2.5e-4,
  gpus: 1, time_min: 120, sweep_lr: [], sweep_steps: [],
}

/** "1e-4, 3e-4" -> [0.0001, 0.0003]; blanks and junk are dropped. */
const parseList = (text: string) =>
  text.split(/[\s,]+/).map(Number).filter((n) => Number.isFinite(n) && n > 0)

function statusTone(status: string): string {
  if (status === 'completed' || status === 'succeeded') return 'chip-ok'
  if (status === 'failed' || status === 'error' || status === 'cancelled') return 'chip-danger'
  if (ACTIVE.has(status)) return 'chip-accent'
  return ''
}

export function ClusterDialog({ onClose, onOpenSettings, onNotify }: Props) {
  const [status, setStatus] = useState<ClusterStatus | null>(null)
  const [jobs, setJobs] = useState<ClusterJob[]>([])
  const [nodeCount, setNodeCount] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [form, setForm] = useState<FineTuneRequest>(DEFAULT_FORM)
  const [sweepLr, setSweepLr] = useState('')
  const [sweepSteps, setSweepSteps] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [selected, setSelected] = useState<ClusterJob | null>(null)
  const [log, setLog] = useState('')
  const logOffset = useRef(0)
  const logBox = useRef<HTMLPreElement>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const s = await api.cluster.status()
      setStatus(s)
      if (s.checkpoints?.length && !s.checkpoints.includes(DEFAULT_FORM.base)) {
        setForm((f) => ({ ...f, base: s.checkpoints![0] }))
      }
      if (s.configured) {
        const [j, n] = await Promise.all([api.cluster.jobs(), api.cluster.nodes()])
        setJobs(j.jobs)
        setNodeCount(n.nodes.length)
      }
    } catch (err) {
      const message = (err as Error).message
      setError(message === 'Not Found'
        ? 'This Shree server was started before the cluster feature was added. Restart Shree (python scripts/launch.py) and open this again.'
        : message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // Stream the selected job's log while it runs; one final read after it stops.
  useEffect(() => {
    if (!selected) return
    let stopped = false
    let timer = 0
    logOffset.current = 0
    const tick = async () => {
      try {
        const r = await api.cluster.log(selected.id, logOffset.current)
        if (stopped) return
        logOffset.current = r.offset
        if (r.text) setLog((prev) => (prev + r.text).slice(-MAX_LOG_CHARS))
        if (r.status !== selected.status) {
          setJobs((all) => all.map((j) => (j.id === selected.id ? { ...j, status: r.status } : j)))
        }
        if (ACTIVE.has(r.status)) timer = window.setTimeout(tick, LOG_POLL_MS)
      } catch (err) {
        if (!stopped) setLog((prev) => `${prev}\n[log unavailable: ${(err as Error).message}]`)
      }
    }
    void tick()
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [selected])

  useEffect(() => {
    logBox.current?.scrollTo({ top: logBox.current.scrollHeight })
  }, [log])

  const select = (job: ClusterJob | null) => {
    setLog('')
    setSelected(job)
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      const r = await api.cluster.finetune({ ...form, sweep_lr: parseList(sweepLr), sweep_steps: parseList(sweepSteps) })
      setJobs((all) => [...r.jobs, ...all])
      select(r.jobs[0] ?? null)
      onNotify(r.jobs.length > 1 ? `Submitted ${r.jobs.length} sweep jobs` : 'Fine-tune job submitted', 'ok')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const cancel = async (job: ClusterJob) => {
    if (!window.confirm(`Cancel "${job.name}"? The run stops and cannot be resumed.`)) return
    try {
      await api.cluster.cancel(job.id)
      setJobs((all) => all.map((j) => (j.id === job.id ? { ...j, status: 'cancelled' } : j)))
      onNotify('Job cancelled', 'info')
    } catch (err) {
      onNotify((err as Error).message, 'danger')
    }
  }

  const download = async (job: ClusterJob) => {
    try {
      const r = await api.cluster.download(job.id)
      onNotify(`Outputs saved to ${r.path}`, 'ok')
    } catch (err) {
      onNotify((err as Error).message, 'danger')
    }
  }

  const set = <K extends keyof FineTuneRequest>(key: K, value: FineTuneRequest[K]) =>
    setForm((f) => ({ ...f, [key]: value }))

  const sweepCount = Math.max(1, parseList(sweepLr).length) * Math.max(1, parseList(sweepSteps).length)

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog dialog-wide" role="dialog" aria-modal="true" aria-labelledby="cluster-title">
        <div className="dialog-head">
          <span className="dialog-title" id="cluster-title"><IconCluster size={15} /> GPU cluster</span>
          {status?.configured && (
            <span className="cluster-meta">
              {status.url?.replace(/^https?:\/\//, '')}
              {nodeCount !== null && <> · {nodeCount} {nodeCount === 1 ? 'node' : 'nodes'}</>}
            </span>
          )}
          <span className="spacer" />
          <button className="btn btn-ghost btn-icon" onClick={() => void refresh()} aria-label="Refresh" title="Refresh">
            <IconRefresh size={14} />
          </button>
          <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Close">
            <IconClose size={14} />
          </button>
        </div>

        <div className="dialog-body">
          {loading && !status && <div className="empty">Checking the cluster…</div>}

          {status && !status.installed && (
            <div className="cluster-setup">
              <h3>Install the ClusterOne SDK</h3>
              <p>The server can’t find the <code>clusterone</code> package. From the Shree folder run:</p>
              <pre className="cluster-cmd">pip install -e ../clausterone</pre>
              <p>Then restart Shree.</p>
            </div>
          )}

          {status?.installed && !status.configured && (
            <div className="cluster-setup">
              <h3>Connect your lab cluster</h3>
              <ol>
                <li>Open Settings and pick <strong>ClusterOne</strong> as the provider.</li>
                <li>Base URL: <code>https://&lt;your-cluster&gt;/api/v1/openai/v1</code></li>
                <li>Paste your ClusterOne API key and scan.</li>
              </ol>
              <p>The same key lets you chat with cluster models and run fine-tunes here.</p>
              <button className="btn btn-primary" onClick={onOpenSettings}>Open settings</button>
            </div>
          )}

          {error && <div className="cluster-error" role="alert">{error}</div>}

          {status?.configured && (
            <div className="cluster-grid">
              <form className="cluster-form" onSubmit={(e) => void submit(e)}>
                <h3>Fine-tune Shree</h3>
                <p className="cluster-lead">
                  Uploads the training code, datasets and one base checkpoint, then runs
                  <code> training/train_coding.py</code> on a cluster GPU.
                </p>
                <div className="field">
                  <label htmlFor="c-name">Run name</label>
                  <input id="c-name" value={form.name} maxLength={80} onChange={(e) => set('name', e.target.value)} required />
                </div>
                <div className="field-row">
                  <div className="field">
                    <label htmlFor="c-base">Base checkpoint</label>
                    <select id="c-base" value={form.base} onChange={(e) => set('base', e.target.value)}>
                      {(status.checkpoints ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="field">
                    <label htmlFor="c-config">Model size</label>
                    <select id="c-config" value={form.config} onChange={(e) => set('config', e.target.value as FineTuneRequest['config'])}>
                      <option value="small">Small</option>
                      <option value="medium">Medium</option>
                      <option value="large">Large</option>
                    </select>
                  </div>
                </div>
                <div className="field-row">
                  <div className="field">
                    <label htmlFor="c-steps">Steps</label>
                    <input id="c-steps" type="number" min={10} max={200000} value={form.steps}
                      onChange={(e) => set('steps', Number(e.target.value))} />
                  </div>
                  <div className="field">
                    <label htmlFor="c-lr">Learning rate</label>
                    <input id="c-lr" type="number" step="any" min={0} max={0.1} value={form.lr}
                      onChange={(e) => set('lr', Number(e.target.value))} />
                  </div>
                </div>
                <div className="field-row">
                  <div className="field">
                    <label htmlFor="c-gpus">GPUs</label>
                    <input id="c-gpus" type="number" min={1} max={8} value={form.gpus}
                      onChange={(e) => set('gpus', Number(e.target.value))} />
                  </div>
                  <div className="field">
                    <label htmlFor="c-time">Time limit (min)</label>
                    <input id="c-time" type="number" min={5} max={10080} value={form.time_min}
                      onChange={(e) => set('time_min', Number(e.target.value))} />
                  </div>
                </div>
                <details className="cluster-sweep">
                  <summary>Sweep several settings</summary>
                  <div className="field">
                    <label htmlFor="c-slr">Learning rates</label>
                    <input id="c-slr" placeholder="1e-4, 3e-4" value={sweepLr} onChange={(e) => setSweepLr(e.target.value)} />
                  </div>
                  <div className="field">
                    <label htmlFor="c-ssteps">Steps</label>
                    <input id="c-ssteps" placeholder="300, 600" value={sweepSteps} onChange={(e) => setSweepSteps(e.target.value)} />
                    <div className="hint">One job per combination, up to 6.</div>
                  </div>
                </details>
                <button className="btn btn-primary btn-block" type="submit" disabled={submitting || sweepCount > 6}>
                  {submitting ? 'Uploading…' : sweepCount > 1 ? `Start ${sweepCount} runs` : 'Start fine-tune'}
                </button>
              </form>

              <section className="cluster-jobs" aria-label="Your jobs">
                <h3>Your jobs</h3>
                {jobs.length === 0 && <div className="empty">No jobs yet. Start a fine-tune to see it here.</div>}
                <ul className="job-list">
                  {jobs.map((job) => (
                    <li key={job.id} className={`job${selected?.id === job.id ? ' selected' : ''}`}>
                      <button className="job-main" onClick={() => select(job)}>
                        <span className="job-name truncate">{job.name}</span>
                        <span className={`chip ${statusTone(job.status)}`}>{job.status.replace('_', ' ')}</span>
                      </button>
                      <span className="job-actions">
                        {ACTIVE.has(job.status) ? (
                          <button className="btn btn-ghost btn-icon btn-sm" onClick={() => void cancel(job)} aria-label={`Cancel ${job.name}`} title="Cancel">
                            <IconStop size={13} />
                          </button>
                        ) : (
                          <button className="btn btn-ghost btn-icon btn-sm" onClick={() => void download(job)} aria-label={`Download outputs of ${job.name}`} title="Download outputs">
                            <IconDownload size={13} />
                          </button>
                        )}
                        <a className="btn btn-ghost btn-icon btn-sm" href={job.url} target="_blank" rel="noopener noreferrer" aria-label={`Open ${job.name} on the cluster`} title="Open on the cluster">
                          <IconExternal size={13} />
                        </a>
                      </span>
                    </li>
                  ))}
                </ul>
                {selected && (
                  <div className="job-log">
                    <div className="job-log-head">Log · {selected.name}</div>
                    <pre ref={logBox} aria-live="polite">{log || 'Waiting for output…'}</pre>
                  </div>
                )}
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
