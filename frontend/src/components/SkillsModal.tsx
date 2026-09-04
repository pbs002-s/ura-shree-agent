import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { IconBolt, IconClose, IconPlus, IconTrash } from '../lib/icons'
import type { Skill } from '../types'

interface SkillsModalProps {
  open?: boolean
  onClose: () => void
  onNotify?: (message: string, tone?: 'ok' | 'danger' | 'info') => void
}

export function SkillsModal({ open = true, onClose, onNotify }: SkillsModalProps) {
  const [skills, setSkills] = useState<Skill[]>([])
  const [loading, setLoading] = useState(false)
  const [showAddForm, setShowAddForm] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [prompt, setPrompt] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const loadSkills = async () => {
    setLoading(true)
    try {
      const data = await api.skills()
      setSkills(data)
    } catch (err) {
      setError((err as Error).message || 'Failed to load skills')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open) {
      loadSkills()
      setError('')
      setShowAddForm(false)
    }
  }, [open])

  if (!open) return null

  const handleToggle = async (skill: Skill) => {
    try {
      const updated = await api.toggleSkill(skill.id, !skill.enabled)
      setSkills((prev) => prev.map((s) => (s.id === skill.id ? updated : s)))
      onNotify?.(`Skill "${skill.name}" ${updated.enabled ? 'enabled' : 'disabled'}`, 'ok')
    } catch (err) {
      const msg = (err as Error).message || 'Failed to toggle skill'
      setError(msg)
      onNotify?.(msg, 'danger')
    }
  }

  const handleDelete = async (skillId: string) => {
    if (!window.confirm('Delete this custom skill?')) return
    try {
      await api.deleteSkill(skillId)
      setSkills((prev) => prev.filter((s) => s.id !== skillId))
      onNotify?.('Skill deleted', 'ok')
    } catch (err) {
      const msg = (err as Error).message || 'Failed to delete skill'
      setError(msg)
      onNotify?.(msg, 'danger')
    }
  }

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim() || !prompt.trim() || saving) return
    setSaving(true)
    setError('')
    try {
      const newSkill = await api.addSkill(name, description, prompt)
      setSkills((prev) => [...prev, newSkill])
      setName('')
      setDescription('')
      setPrompt('')
      setShowAddForm(false)
      onNotify?.(`Added new skill "${newSkill.name}"`, 'ok')
    } catch (err) {
      const msg = (err as Error).message || 'Failed to add skill'
      setError(msg)
      onNotify?.(msg, 'danger')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card skills-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <IconBolt size={18} className="accent" />
            <div>
              <div className="modal-title">Agent Skills & Workflows</div>
              <div className="modal-subtitle faint">
                Active skills steer Shree's specialized domain behaviors and system prompts.
              </div>
            </div>
          </div>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose} title="Close skills">
            <IconClose size={15} />
          </button>
        </div>

        {error && <div className="banner banner-danger">{error}</div>}

        <div className="skills-body">
          <div className="skills-toolbar">
            <span className="section-label">Configured Skills ({skills.length})</span>
            <button
              className="btn btn-primary btn-sm"
              onClick={() => setShowAddForm(!showAddForm)}
            >
              <IconPlus size={13} />
              <span>{showAddForm ? 'Cancel' : 'Add Custom Skill'}</span>
            </button>
          </div>

          {showAddForm && (
            <form className="skills-add-form" onSubmit={handleCreate}>
              <div className="field">
                <label className="field-label">Skill Name *</label>
                <input
                  type="text"
                  placeholder="e.g. Flutter Mobile Engineer or Docker & DevOps"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>

              <div className="field">
                <label className="field-label">Brief Description</label>
                <input
                  type="text"
                  placeholder="Short summary of when this skill applies"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>

              <div className="field">
                <label className="field-label">Prompt Instructions *</label>
                <textarea
                  rows={4}
                  placeholder="Specific rules, principles, and guidelines injected into Shree's prompt..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  required
                />
              </div>

              <div className="skills-form-actions">
                <button type="submit" className="btn btn-primary btn-sm" disabled={saving}>
                  {saving ? 'Creating…' : 'Save Skill'}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => setShowAddForm(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}

          {loading ? (
            <div className="code-editor-loading faint">Loading skills…</div>
          ) : (
            <div className="skills-grid">
              {skills.map((skill) => (
                <div
                  key={skill.id}
                  className={`skill-card ${skill.enabled ? 'active' : 'inactive'}`}
                >
                  <div className="skill-card-head">
                    <div className="skill-card-info">
                      <span className="skill-name">{skill.name}</span>
                      {skill.built_in && <span className="chip chip-dim">Built-in</span>}
                    </div>
                    <div className="skill-card-actions">
                      <label className="switch" title={skill.enabled ? 'Disable skill' : 'Enable skill'}>
                        <input
                          type="checkbox"
                          checked={skill.enabled}
                          onChange={() => handleToggle(skill)}
                        />
                        <span className="slider round" />
                      </label>
                      {!skill.built_in && (
                        <button
                          className="btn btn-ghost btn-sm btn-icon"
                          onClick={() => handleDelete(skill.id)}
                          title="Delete custom skill"
                        >
                          <IconTrash size={13} />
                        </button>
                      )}
                    </div>
                  </div>

                  {skill.description && (
                    <div className="skill-description faint">{skill.description}</div>
                  )}

                  <div className="skill-prompt-preview mono">{skill.prompt}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
