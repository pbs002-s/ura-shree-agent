export type Theme = 'light' | 'dark' | 'system'

export interface ProviderSpec {
  id: string
  label: string
  protocol: string
  base_url: string
  docs_url: string
  key_url: string
  key_prefix: string
  requires_key: boolean
  supports_tools: boolean
  fallback_models: string[]
  notes: string
}

export interface ModelInfo {
  id: string
  label: string
  context_window: number
  supports_tools: boolean
  owned_by: string
}

export interface ProviderState {
  provider: string
  has_key: boolean
  key_preview: string
  base_url: string
  models: ModelInfo[]
  selected_model: string
}

export interface ActiveSettings {
  provider: string
  model: string
  temperature: number
  max_tokens: number
  use_tools: boolean
  auto_approve: boolean
}

export interface LocalSettings {
  checkpoint: string
  device: string
  quantize: boolean
  compile: boolean
}

export interface AppSettings {
  version: number
  active: ActiveSettings
  theme: Theme
  local: LocalSettings
  providers: Record<string, ProviderState>
}

export interface Status {
  status: string
  version: string
  workspace: string
  platform: string
  active: ActiveSettings
  local_model: {
    loaded: boolean
    error: string
    checkpoint?: string
    parameters?: number
    architecture?: Record<string, number | string>
    memory?: Record<string, number>
    last_generation?: Record<string, number>
  }
  runtime: {
    device?: string
    device_name?: string
    dtype?: string
    cpu_threads?: number
    physical_cores?: number
    logical_cores?: number
    total_ram_gb?: number
    available_ram_gb?: number
    total_vram_gb?: number
    notes?: string[]
  }
  hardware: {
    cuda_available: boolean
    device: string
    allocated_mb: number
    reserved_mb: number
    total_mb: number
    free_mb: number
    process_rss_mb: number
    system_used_mb: number
    system_total_mb: number
  }
  time_machine: { head: string | null; store_bytes: number }
}

export interface TreeNode {
  name: string
  path: string
  isDir: boolean
  size?: number
  children?: TreeNode[]
}

export interface Snapshot {
  id: string
  parent_id: string | null
  label: string
  kind: string
  created_at: number
  file_count: number
  total_bytes: number
  meta: Record<string, unknown>
  is_branch_point?: boolean
  unchanged?: boolean
}

export interface DiffFile {
  path: string
  status: 'added' | 'removed' | 'modified'
  binary: boolean
  diff: string
  additions: number
  deletions: number
}

export interface SnapshotDiff {
  from: string
  to: string
  summary: { added: number; removed: number; modified: number; truncated: boolean }
  files: DiffFile[]
}

export interface Skill {
  id: string
  name: string
  description: string
  prompt: string
  enabled: boolean
  built_in?: boolean
}

export interface ToolRun {
  id: string
  name: string
  arguments: Record<string, unknown>
  mutating: boolean
  status: 'running' | 'ok' | 'failed' | 'awaiting_approval'
  text: string
  data: Record<string, unknown>
  needs_approval?: boolean
}

export type BlockKind = 'text' | 'thinking' | 'tool' | 'error'

export interface Block {
  kind: BlockKind
  text: string
  tool?: ToolRun
}

export interface Turn {
  id: string
  role: 'user' | 'assistant'
  text: string
  blocks: Block[]
  attachments?: { name: string; size: number }[]
  streaming?: boolean
  meta?: { provider?: string; model?: string; durationMs?: number; usage?: Record<string, number> }
}

export interface TerminalLine {
  kind: 'command' | 'output' | 'error' | 'note'
  text: string
}

export interface Attachment {
  name: string
  path: string
  size: number
  content: string
  base64: string
  isText: boolean
}
