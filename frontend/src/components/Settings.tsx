import { useEffect, useMemo, useState } from 'react'
import { api, formatBytes, formatNumber } from '../lib/api'
import {
  IconAlert, IconCheck, IconClose, IconCpu, IconKey, IconRefresh, IconSearch, IconTrash,
} from '../lib/icons'
import type { AppSettings, ModelInfo, ProviderSpec, Status, Theme } from '../types'

type Section = 'models' | 'behaviour' | 'local' | 'appearance' | 'about'

const NO_MODELS: ModelInfo[] = []

const SECTIONS: { id: Section; label: string }[] = [
  { id: 'models', label: 'Models and keys' },
  { id: 'behaviour', label: 'Agent behaviour' },
  { id: 'local', label: 'Local model' },
  { id: 'appearance', label: 'Appearance' },
  { id: 'about', label: 'About' },
]

interface SettingsProps {
  settings: AppSettings
  status: Status | null
  theme: Theme
  onThemeChange: (theme: Theme) => void
  onSettingsChange: (settings: AppSettings) => void
  onNotify: (message: string, tone?: 'ok' | 'danger') => void
  onClose: () => void
}

export function SettingsDialog({
  settings, status, theme, onThemeChange, onSettingsChange, onNotify, onClose,
}: SettingsProps) {
  const [section, setSection] = useState<Section>('models')
  const [specs, setSpecs] = useState<ProviderSpec[]>([])
  const [providerId, setProviderId] = useState(settings.active.provider || 'local')
  const [keyDraft, setKeyDraft] = useState('')
  const [baseDraft, setBaseDraft] = useState('')
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState('')
  const [filter, setFilter] = useState('')

  useEffect(() => {
    api.providers()
      .then((data) => {
        setSpecs(data.providers)
        const current = data.providers.find((s) => s.id === providerId)
        setBaseDraft(settings.providers[providerId]?.base_url ?? current?.base_url ?? '')
      })
      .catch(() => undefined)
    // Runs once: the catalogue is static for the life of the dialog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const spec = useMemo(() => specs.find((s) => s.id === providerId), [specs, providerId])
  const state = settings.providers[providerId]
  // A stable reference, so the filter memo below does not recompute every render.
  const models: ModelInfo[] = state?.models ?? NO_MODELS

  // Switching provider clears the form. Done here rather than in an effect:
  // the change is caused by this event, so there is nothing to synchronise.
  const switchProvider = (next: string) => {
    const nextSpec = specs.find((s) => s.id === next)
    setProviderId(next)
    setKeyDraft('')
    setBaseDraft(settings.providers[next]?.base_url ?? nextSpec?.base_url ?? '')
    setScanError('')
    setFilter('')
  }

  const filtered = useMemo(() => {
    if (!filter.trim()) return models
    const needle = filter.toLowerCase()
    return models.filter((m) => m.id.toLowerCase().includes(needle) || m.label.toLowerCase().includes(needle))
  }, [models, filter])

  const scan = async () => {
    setScanning(true)
    setScanError('')
    try {
      const result = await api.scanModels(providerId, keyDraft || undefined, baseDraft || undefined)
      const fresh = await api.settings()
      onSettingsChange(fresh)
      setKeyDraft('')
      if (result.error) setScanError(result.error)
      if (result.models.length) {
        onNotify(
          `Found ${result.models.length} model(s)` +
          (result.source === 'fallback' ? ' from the known list; the live scan failed.' : '.'),
          result.source === 'fallback' ? 'danger' : 'ok',
        )
      } else {
        onNotify('No models found. Check the key and base URL.', 'danger')
      }
    } catch (err) {
      setScanError((err as Error).message)
      onNotify((err as Error).message, 'danger')
    } finally {
      setScanning(false)
    }
  }

  const choose = async (modelId: string) => {
    try {
      const fresh = await api.selectModel(providerId, modelId)
      onSettingsChange(fresh)
      onNotify(`Active model: ${modelId}`, 'ok')
    } catch (err) {
      onNotify((err as Error).message, 'danger')
    }
  }

  const forget = async () => {
    if (!window.confirm(`Forget the stored key and models for ${spec?.label ?? providerId}?`)) return
    await api.forgetProvider(providerId)
    onSettingsChange(await api.settings())
    onNotify('Credentials removed.', 'ok')
  }

  const patch = async (sectionName: string, values: Record<string, unknown>) => {
    try {
      onSettingsChange(await api.updateSettings(sectionName, values))
    } catch (err) {
      onNotify((err as Error).message, 'danger')
    }
  }

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label="Settings">
        <div className="dialog-head">
          <span className="dialog-title">Settings</span>
          <span className="spacer" />
          <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Close settings">
            <IconClose size={15} />
          </button>
        </div>

        <div className="dialog-body">
          <div className="settings-grid">
            <nav className="settings-nav">
              {SECTIONS.map((item) => (
                <button
                  key={item.id}
                  className={`settings-item${section === item.id ? ' active' : ''}`}
                  onClick={() => setSection(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </nav>

            <div className="settings-main">
              {section === 'models' && (
                <>
                  <div className="field">
                    <label htmlFor="provider">Provider</label>
                    <select
                      id="provider"
                      value={providerId}
                      onChange={(e) => switchProvider(e.target.value)}
                    >
                      {specs.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.label}
                          {settings.providers[s.id]?.has_key ? '  (key saved)' : ''}
                        </option>
                      ))}
                    </select>
                    {spec?.notes && <div className="hint">{spec.notes}</div>}
                  </div>

                  {spec?.requires_key && (
                    <div className="field">
                      <label htmlFor="apikey">
                        API key
                        {state?.has_key && (
                          <span className="chip chip-ok" style={{ marginLeft: 8 }}>
                            <IconCheck size={10} /> saved {state.key_preview}
                          </span>
                        )}
                      </label>
                      <input
                        id="apikey"
                        type="password"
                        value={keyDraft}
                        autoComplete="off"
                        spellCheck={false}
                        placeholder={state?.has_key ? 'Paste a new key to replace the saved one' : `Paste your key${spec.key_prefix ? ` (starts with ${spec.key_prefix})` : ''}`}
                        onChange={(e) => setKeyDraft(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && void scan()}
                      />
                      <div className="hint">
                        Stored on this machine in <code>.shree/settings.json</code> and never sent
                        anywhere except {spec.label}.
                        {spec.key_url && (
                          <> <a href={spec.key_url} target="_blank" rel="noreferrer noopener">Get a key</a>.</>
                        )}
                      </div>
                    </div>
                  )}

                  {(spec?.id === 'custom' || spec?.id === 'ollama' || spec?.id === 'lmstudio') && (
                    <div className="field">
                      <label htmlFor="baseurl">Base URL</label>
                      <input
                        id="baseurl"
                        value={baseDraft}
                        placeholder="http://localhost:11434/v1"
                        onChange={(e) => setBaseDraft(e.target.value)}
                      />
                    </div>
                  )}

                  <div className="row" style={{ marginBottom: 14 }}>
                    <button className="btn btn-primary" onClick={() => void scan()} disabled={scanning}>
                      {scanning ? <IconRefresh size={13} /> : <IconSearch size={13} />}
                      {scanning ? 'Scanning…' : 'Scan available models'}
                    </button>
                    {state?.has_key && (
                      <button className="btn btn-danger" onClick={() => void forget()}>
                        <IconTrash size={13} /> Forget key
                      </button>
                    )}
                  </div>

                  {scanError && (
                    <div className="banner banner-warn" style={{ marginBottom: 12 }}>
                      <IconAlert size={14} style={{ flex: 'none', marginTop: 1 }} />
                      <span>{scanError}</span>
                    </div>
                  )}

                  {models.length > 0 && (
                    <div className="field">
                      <label>
                        {models.length} model{models.length === 1 ? '' : 's'} available
                      </label>
                      {models.length > 8 && (
                        <input
                          value={filter}
                          onChange={(e) => setFilter(e.target.value)}
                          placeholder="Filter models"
                          style={{ marginBottom: 6, height: 28 }}
                        />
                      )}
                      <div className="model-list">
                        {filtered.map((model) => {
                          const active =
                            settings.active.provider === providerId && settings.active.model === model.id
                          return (
                            <button
                              key={model.id}
                              className={`model-row${active ? ' selected' : ''}`}
                              onClick={() => void choose(model.id)}
                            >
                              {active ? <IconCheck size={13} /> : <span style={{ width: 13 }} />}
                              <span className="truncate">{model.label}</span>
                              {model.label !== model.id && (
                                <span className="id faint truncate">{model.id}</span>
                              )}
                              <span className="spacer" />
                              {model.context_window > 0 && (
                                <span className="faint" style={{ fontSize: 'var(--fs-xs)' }}>
                                  {formatNumber(model.context_window)} ctx
                                </span>
                              )}
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </>
              )}

              {section === 'behaviour' && (
                <>
                  <div className="field">
                    <label htmlFor="temp">Temperature · {settings.active.temperature}</label>
                    <input
                      id="temp"
                      type="range"
                      min={0}
                      max={1.5}
                      step={0.05}
                      value={settings.active.temperature}
                      onChange={(e) => void patch('active', { temperature: Number(e.target.value) })}
                    />
                    <div className="hint">Lower is more deterministic. 0.2 or below suits code edits.</div>
                  </div>

                  <div className="field">
                    <label htmlFor="maxtok">Maximum reply tokens</label>
                    <input
                      id="maxtok"
                      type="number"
                      min={256}
                      max={32000}
                      step={256}
                      value={settings.active.max_tokens}
                      onChange={(e) => void patch('active', { max_tokens: Number(e.target.value) })}
                    />
                  </div>

                  <div className="field">
                    <label className="row" style={{ cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        style={{ width: 'auto' }}
                        checked={settings.active.use_tools}
                        onChange={(e) => void patch('active', { use_tools: e.target.checked })}
                      />
                      Give the model tools
                    </label>
                    <div className="hint">
                      Off means plain conversation: no file reads, edits or commands.
                    </div>
                  </div>

                  <div className="field">
                    <label className="row" style={{ cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        style={{ width: 'auto' }}
                        checked={settings.active.auto_approve}
                        onChange={(e) => void patch('active', { auto_approve: e.target.checked })}
                      />
                      Run file writes and commands without asking
                    </label>
                    <div className="hint">
                      Turn this off to make the agent refuse anything that changes the workspace.
                      Every edit is snapshotted either way.
                    </div>
                  </div>
                </>
              )}

              {section === 'local' && (
                <>
                  <div className="field">
                    <label htmlFor="device">Device</label>
                    <select
                      id="device"
                      value={settings.local.device || ''}
                      onChange={(e) => void patch('local', { device: e.target.value })}
                    >
                      <option value="">Automatic</option>
                      <option value="cuda">CUDA (GPU)</option>
                      <option value="cpu">CPU</option>
                    </select>
                  </div>

                  <div className="field">
                    <label className="row" style={{ cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        style={{ width: 'auto' }}
                        checked={settings.local.quantize}
                        onChange={(e) => void patch('local', { quantize: e.target.checked })}
                      />
                      Quantise to int8 on CPU
                    </label>
                    <div className="hint">
                      Roughly four times less resident memory than float32, with a small accuracy
                      cost. Ignored on GPU.
                    </div>
                  </div>

                  <div className="field">
                    <label className="row" style={{ cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        style={{ width: 'auto' }}
                        checked={settings.local.compile}
                        onChange={(e) => void patch('local', { compile: e.target.checked })}
                      />
                      Compile the model graph
                    </label>
                    <div className="hint">
                      The first reply is slow while it compiles, later ones are faster.
                    </div>
                  </div>

                  {status?.local_model.loaded && status.local_model.architecture && (
                    <div style={{ marginTop: 18 }}>
                      <div className="section-label" style={{ marginBottom: 8 }}>Loaded checkpoint</div>
                      <dl className="kv">
                        <dt>Checkpoint</dt><dd className="mono">{status.local_model.checkpoint}</dd>
                        <dt>Parameters</dt><dd>{formatNumber(status.local_model.parameters ?? 0)}</dd>
                        <dt>Layers / heads</dt>
                        <dd>
                          {String(status.local_model.architecture.layers)} / {String(status.local_model.architecture.heads)}
                        </dd>
                        <dt>Context</dt><dd>{formatNumber(Number(status.local_model.architecture.context))}</dd>
                        <dt>Positions</dt><dd>{String(status.local_model.architecture.pos_encoding)}</dd>
                        <dt>Feed-forward</dt><dd>{String(status.local_model.architecture.ffn)}</dd>
                        <dt>Weights</dt><dd>{status.local_model.memory?.total_mb} MB</dd>
                        <dt>Full KV cache</dt><dd>{status.local_model.memory?.kv_cache_full_mb} MB</dd>
                      </dl>
                    </div>
                  )}
                  {status && !status.local_model.loaded && status.local_model.error && (
                    <div className="banner banner-warn" style={{ marginTop: 12 }}>
                      <IconAlert size={14} style={{ flex: 'none', marginTop: 1 }} />
                      <span>{status.local_model.error}</span>
                    </div>
                  )}
                </>
              )}

              {section === 'appearance' && (
                <div className="field">
                  <label>Theme</label>
                  <div className="seg">
                    {(['light', 'dark', 'system'] as Theme[]).map((option) => (
                      <button
                        key={option}
                        className={theme === option ? 'active' : ''}
                        onClick={() => onThemeChange(option)}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                  <div className="hint">System follows your operating system setting.</div>
                </div>
              )}

              {section === 'about' && status && (
                <>
                  <div className="section-label" style={{ marginBottom: 8 }}>Runtime</div>
                  <dl className="kv" style={{ marginBottom: 18 }}>
                    <dt>Version</dt><dd>URA-Shree {status.version}</dd>
                    <dt>Workspace</dt><dd className="mono truncate">{status.workspace}</dd>
                    <dt>Platform</dt><dd>{status.platform}</dd>
                    <dt>Compute</dt><dd>{status.runtime.device_name} ({status.runtime.device})</dd>
                    <dt>Precision</dt><dd>{status.runtime.dtype}</dd>
                    <dt>CPU threads</dt>
                    <dd>
                      {status.runtime.cpu_threads} of {status.runtime.logical_cores} logical,{' '}
                      {status.runtime.physical_cores} physical
                    </dd>
                    <dt>System RAM</dt>
                    <dd>{status.runtime.available_ram_gb} GB free of {status.runtime.total_ram_gb} GB</dd>
                    {status.hardware.cuda_available && (
                      <>
                        <dt>VRAM</dt>
                        <dd>
                          {(status.hardware.free_mb / 1024).toFixed(1)} GB free of{' '}
                          {(status.hardware.total_mb / 1024).toFixed(1)} GB
                        </dd>
                      </>
                    )}
                    <dt>Snapshot store</dt><dd>{formatBytes(status.time_machine.store_bytes)}</dd>
                  </dl>

                  {!!status.runtime.notes?.length && (
                    <>
                      <div className="section-label" style={{ marginBottom: 6 }}>Optimisations applied</div>
                      <ul className="muted" style={{ margin: '0 0 18px', paddingLeft: 18, fontSize: 'var(--fs-sm)' }}>
                        {status.runtime.notes.map((note) => <li key={note}>{note}</li>)}
                      </ul>
                    </>
                  )}

                  <div className="section-label" style={{ marginBottom: 6 }}>Credits</div>
                  <p className="muted" style={{ margin: 0, fontSize: 'var(--fs-sm)' }}>
                    Built by{' '}
                    <a href="https://github.com/pbs002-s" target="_blank" rel="noreferrer noopener">
                      github.com/pbs002-s
                    </a>
                    . The model, tokenizer, training loop, agent and interface are all in this
                    repository.
                  </p>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="dialog-foot">
          <span className="chip">
            <IconKey size={11} />
            {Object.values(settings.providers).filter((p) => p.has_key).length} key(s) saved
          </span>
          <span className="chip">
            <IconCpu size={11} />
            {status?.runtime.device ?? 'unknown'}
          </span>
          <span className="spacer" />
          <button className="btn" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  )
}
