/**
 * A small markdown renderer built for streaming.
 *
 * Written by hand rather than pulled in as a dependency for one reason: a
 * general parser sees a half-arrived fence as broken input, and the reply is
 * always half-arrived here. This one treats an unterminated ``` as a code block
 * still being written, so code renders as code from the first line instead of
 * flickering into place when the closing fence lands.
 *
 * Output is React elements, never dangerouslySetInnerHTML - model output is
 * untrusted text and must never reach the DOM as markup.
 */

import { Fragment, type ReactNode } from 'react'

type Token =
  | { type: 'code'; lang: string; body: string; open: boolean }
  | { type: 'heading'; level: number; body: string }
  | { type: 'list'; ordered: boolean; items: string[]; start: number }
  | { type: 'quote'; body: string }
  | { type: 'rule' }
  | { type: 'table'; header: string[]; rows: string[][] }
  | { type: 'para'; body: string }

const FENCE = /^\s*(`{3,}|~{3,})\s*([\w+-]*)\s*$/
const HEADING = /^(#{1,6})\s+(.*)$/
const UL = /^\s*[-*+]\s+(.*)$/
const OL = /^\s*(\d+)[.)]\s+(.*)$/
const QUOTE = /^\s*>\s?(.*)$/
const RULE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/

function splitRow(line: string): string[] {
  return line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim())
}

function tokenize(source: string): Token[] {
  const lines = source.split('\n')
  const tokens: Token[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    const fence = line.match(FENCE)
    if (fence) {
      const marker = fence[1]
      const lang = fence[2] || ''
      const body: string[] = []
      i += 1
      let closed = false
      while (i < lines.length) {
        if (lines[i].trimStart().startsWith(marker)) {
          closed = true
          i += 1
          break
        }
        body.push(lines[i])
        i += 1
      }
      tokens.push({ type: 'code', lang, body: body.join('\n'), open: !closed })
      continue
    }

    if (!line.trim()) {
      i += 1
      continue
    }

    if (RULE.test(line)) {
      tokens.push({ type: 'rule' })
      i += 1
      continue
    }

    const heading = line.match(HEADING)
    if (heading) {
      tokens.push({ type: 'heading', level: heading[1].length, body: heading[2] })
      i += 1
      continue
    }

    // A table needs a header row followed by a separator row.
    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const header = splitRow(line)
      i += 2
      const rows: string[][] = []
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(splitRow(lines[i]))
        i += 1
      }
      tokens.push({ type: 'table', header, rows })
      continue
    }

    if (UL.test(line) || OL.test(line)) {
      const ordered = OL.test(line)
      const start = ordered ? Number(line.match(OL)![1]) : 1
      const items: string[] = []
      while (i < lines.length) {
        const ul = lines[i].match(UL)
        const ol = lines[i].match(OL)
        if (ordered && ol) items.push(ol[2])
        else if (!ordered && ul) items.push(ul[1])
        else if (items.length && /^\s{2,}\S/.test(lines[i])) {
          // Continuation of the previous item.
          items[items.length - 1] += ` ${lines[i].trim()}`
        } else break
        i += 1
      }
      tokens.push({ type: 'list', ordered, items, start })
      continue
    }

    const quote = line.match(QUOTE)
    if (quote) {
      const body: string[] = [quote[1]]
      i += 1
      while (i < lines.length && QUOTE.test(lines[i])) {
        body.push(lines[i].match(QUOTE)![1])
        i += 1
      }
      tokens.push({ type: 'quote', body: body.join('\n') })
      continue
    }

    const body: string[] = [line]
    i += 1
    while (
      i < lines.length && lines[i].trim() &&
      !FENCE.test(lines[i]) && !HEADING.test(lines[i]) &&
      !UL.test(lines[i]) && !OL.test(lines[i]) &&
      !QUOTE.test(lines[i]) && !RULE.test(lines[i])
    ) {
      body.push(lines[i])
      i += 1
    }
    tokens.push({ type: 'para', body: body.join('\n') })
  }

  return tokens
}

/** Inline spans: code, bold, italic, strikethrough, links. */
function inline(text: string, keyPrefix = 'i'): ReactNode[] {
  const nodes: ReactNode[] = []
  const pattern =
    /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\n]+\*)|(_[^_\n]+_)|(~~[^~]+~~)|(\[[^\]]+\]\([^)\s]+\))/g

  let last = 0
  let match: RegExpExecArray | null
  let n = 0

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index))
    const token = match[0]
    const key = `${keyPrefix}-${n++}`

    if (token.startsWith('`')) {
      nodes.push(<code key={key}>{token.slice(1, -1)}</code>)
    } else if (token.startsWith('**') || token.startsWith('__')) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>)
    } else if (token.startsWith('~~')) {
      nodes.push(<del key={key}>{token.slice(2, -2)}</del>)
    } else if (token.startsWith('[')) {
      const link = token.match(/\[([^\]]+)\]\(([^)\s]+)\)/)!
      const href = link[2]
      // Only http(s) links become anchors; a javascript: URL never should.
      if (/^https?:\/\//i.test(href)) {
        nodes.push(
          <a key={key} href={href} target="_blank" rel="noreferrer noopener">{link[1]}</a>,
        )
      } else {
        nodes.push(<Fragment key={key}>{link[1]}</Fragment>)
      }
    } else {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>)
    }
    last = match.index + token.length
  }

  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

/* ── Syntax highlighting ────────────────────────────────────────────────── */

const KEYWORDS: Record<string, string[]> = {
  python: ['def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'import', 'from', 'as',
    'with', 'try', 'except', 'finally', 'raise', 'yield', 'lambda', 'async', 'await', 'pass',
    'break', 'continue', 'global', 'nonlocal', 'assert', 'del', 'in', 'is', 'not', 'and', 'or',
    'True', 'False', 'None', 'self'],
  javascript: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class',
    'extends', 'new', 'this', 'import', 'export', 'from', 'default', 'async', 'await', 'try',
    'catch', 'finally', 'throw', 'typeof', 'instanceof', 'switch', 'case', 'break', 'continue',
    'true', 'false', 'null', 'undefined', 'interface', 'type', 'enum', 'implements', 'public',
    'private', 'readonly', 'as', 'of', 'in'],
  shell: ['if', 'then', 'else', 'fi', 'for', 'in', 'do', 'done', 'while', 'case', 'esac',
    'function', 'return', 'export', 'local', 'echo', 'cd', 'set'],
  rust: ['fn', 'let', 'mut', 'const', 'struct', 'enum', 'impl', 'trait', 'pub', 'use', 'mod',
    'match', 'if', 'else', 'for', 'while', 'loop', 'return', 'self', 'Self', 'where', 'async',
    'await', 'move', 'ref', 'dyn', 'true', 'false'],
  go: ['func', 'var', 'const', 'type', 'struct', 'interface', 'package', 'import', 'return', 'if',
    'else', 'for', 'range', 'switch', 'case', 'defer', 'go', 'chan', 'map', 'nil', 'true', 'false'],
}

const ALIASES: Record<string, keyof typeof KEYWORDS> = {
  py: 'python', python3: 'python',
  js: 'javascript', jsx: 'javascript', ts: 'javascript', tsx: 'javascript',
  typescript: 'javascript', javascript: 'javascript', json: 'javascript',
  sh: 'shell', bash: 'shell', zsh: 'shell', shell: 'shell', powershell: 'shell', ps1: 'shell',
  rust: 'rust', rs: 'rust', go: 'go', golang: 'go',
}

function highlight(code: string, lang: string): ReactNode {
  const family = ALIASES[lang.toLowerCase()]
  if (!family) return code

  const keywords = new Set(KEYWORDS[family])
  const commentStart = family === 'python' || family === 'shell' ? '#' : '//'

  // One pass over strings, comments, numbers, identifiers. A full grammar is
  // overkill for reading a snippet; this is enough to give code shape.
  const pattern = new RegExp(
    `("""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:[^"\\\\\\n]|\\\\.)*"|'(?:[^'\\\\\\n]|\\\\.)*'|\`(?:[^\`\\\\]|\\\\.)*\`)` +
    `|(${commentStart === '#' ? '#' : '\\/\\/'}[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/)` +
    `|(\\b\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b)` +
    `|([A-Za-z_$][\\w$]*)`,
    'g',
  )

  const out: ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  let n = 0

  while ((match = pattern.exec(code)) !== null) {
    if (match.index > last) out.push(code.slice(last, match.index))
    const key = `t${n++}`
    const [token, str, comment, num, word] = match

    if (str) out.push(<span key={key} className="tok-str">{token}</span>)
    else if (comment) out.push(<span key={key} className="tok-com">{token}</span>)
    else if (num) out.push(<span key={key} className="tok-num">{token}</span>)
    else if (word && keywords.has(word)) out.push(<span key={key} className="tok-key">{token}</span>)
    else if (word && /^[A-Z]/.test(word)) out.push(<span key={key} className="tok-typ">{token}</span>)
    else if (word && code[match.index + word.length] === '(') {
      out.push(<span key={key} className="tok-fun">{token}</span>)
    } else out.push(token)

    last = match.index + token.length
  }
  if (last < code.length) out.push(code.slice(last))
  return out
}

export function CodeBlock({ code, lang, open }: { code: string; lang: string; open?: boolean }) {
  const copy = () => {
    void navigator.clipboard?.writeText(code)
  }
  return (
    <div className="code-block">
      <div className="code-head">
        <span className="code-lang">{lang || 'text'}</span>
        {open && <span className="faint">writing…</span>}
        <span className="spacer" />
        <button className="btn btn-ghost btn-sm" onClick={copy} title="Copy code">copy</button>
      </div>
      <pre><code>{highlight(code, lang)}</code></pre>
    </div>
  )
}

export function Markdown({ source }: { source: string }) {
  const tokens = tokenize(source)
  return (
    <div className="md">
      {tokens.map((token, index) => {
        const key = `b${index}`
        switch (token.type) {
          case 'code':
            return <CodeBlock key={key} code={token.body} lang={token.lang} open={token.open} />
          case 'heading': {
            const Tag = `h${Math.min(token.level, 4)}` as 'h1' | 'h2' | 'h3' | 'h4'
            return <Tag key={key}>{inline(token.body, key)}</Tag>
          }
          case 'list':
            return token.ordered ? (
              <ol key={key} start={token.start}>
                {token.items.map((item, n) => <li key={n}>{inline(item, `${key}-${n}`)}</li>)}
              </ol>
            ) : (
              <ul key={key}>
                {token.items.map((item, n) => <li key={n}>{inline(item, `${key}-${n}`)}</li>)}
              </ul>
            )
          case 'quote':
            return <blockquote key={key}>{inline(token.body, key)}</blockquote>
          case 'rule':
            return <hr key={key} />
          case 'table':
            return (
              <table key={key}>
                <thead>
                  <tr>{token.header.map((cell, n) => <th key={n}>{inline(cell, `${key}h${n}`)}</th>)}</tr>
                </thead>
                <tbody>
                  {token.rows.map((row, r) => (
                    <tr key={r}>{row.map((cell, c) => <td key={c}>{inline(cell, `${key}${r}${c}`)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            )
          default:
            return <p key={key}>{inline(token.body, key)}</p>
        }
      })}
    </div>
  )
}
