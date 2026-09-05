/**
 * Session export.
 *
 * The chat lives only in React state, so a session is lost on reload unless
 * someone takes it with them. Markdown is for reading and pasting into an
 * issue; JSON keeps the tool calls and timings intact for replay or analysis.
 */

import type { Turn } from '../types'

function toolLine(name: string, status: string, args: Record<string, unknown>): string {
  const first = args.path ?? args.command ?? args.pattern ?? args.query ?? args.label
  const detail = typeof first === 'string' && first ? ` \`${first}\`` : ''
  return `**${name}**${detail} — ${status}`
}

export function turnsToMarkdown(turns: Turn[], modelLabel: string): string {
  const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19)
  const lines = [
    '# URA-Shree session',
    '',
    `- Exported: ${stamp}`,
    `- Model: ${modelLabel}`,
    `- Turns: ${turns.length}`,
    '',
    '---',
    '',
  ]

  for (const turn of turns) {
    if (turn.role === 'user') {
      lines.push('## You', '', turn.text.trim() || '_(empty)_', '')
      if (turn.attachments?.length) {
        lines.push(`Attached: ${turn.attachments.map((a) => `\`${a.name}\``).join(', ')}`, '')
      }
      continue
    }

    const meta = [turn.meta?.model, turn.meta?.durationMs != null
      ? `${(turn.meta.durationMs / 1000).toFixed(1)}s`
      : null].filter(Boolean).join(' · ')
    lines.push(`## Shree${meta ? ` <sub>${meta}</sub>` : ''}`, '')

    for (const block of turn.blocks) {
      if (block.kind === 'text') lines.push(block.text.trim(), '')
      else if (block.kind === 'thinking') {
        lines.push('<details><summary>Thinking</summary>', '', block.text.trim(), '', '</details>', '')
      } else if (block.kind === 'error') lines.push(`> **Error:** ${block.text.trim()}`, '')
      else if (block.kind === 'tool' && block.tool) {
        const tool = block.tool
        lines.push(toolLine(tool.name, tool.status, tool.arguments), '')
        if (tool.text.trim()) lines.push('```', tool.text.trim(), '```', '')
      }
    }
  }

  return lines.join('\n').replace(/\n{3,}/g, '\n\n')
}

export function turnsToJson(turns: Turn[], modelLabel: string): string {
  return JSON.stringify(
    { exported_at: new Date().toISOString(), model: modelLabel, turns },
    null,
    2,
  )
}

/** Hand the browser a Blob under a filename, then release the object URL. */
export function download(filename: string, content: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function exportSession(turns: Turn[], modelLabel: string, format: 'md' | 'json'): void {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  if (format === 'md') {
    download(`shree-session-${stamp}.md`, turnsToMarkdown(turns, modelLabel), 'text/markdown')
  } else {
    download(`shree-session-${stamp}.json`, turnsToJson(turns, modelLabel), 'application/json')
  }
}
