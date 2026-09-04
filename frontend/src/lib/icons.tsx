/**
 * Inline SVG icons.
 *
 * A local set rather than an icon package: it is a few hundred bytes in the
 * bundle instead of a dependency, every glyph is drawn on the same 24-unit
 * grid with the same stroke weight, and `currentColor` means they follow the
 * theme without any per-icon wiring.
 */

import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function Svg({ size = 16, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  )
}

export const IconChat = (p: IconProps) => (
  <Svg {...p}><path d="M21 12a8 8 0 0 1-11.7 7.1L3 21l1.9-6.3A8 8 0 1 1 21 12Z" /></Svg>
)
export const IconFiles = (p: IconProps) => (
  <Svg {...p}><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" /></Svg>
)
export const IconClock = (p: IconProps) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3.2 1.9" /></Svg>
)
export const IconTerminal = (p: IconProps) => (
  <Svg {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="m7 9 3 3-3 3M13 15h4" /></Svg>
)
export const IconBrain = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 5a3 3 0 0 0-5.9-.7A3 3 0 0 0 4 9.5 3 3 0 0 0 5.5 15 3 3 0 0 0 12 18Z" />
    <path d="M12 5a3 3 0 0 1 5.9-.7A3 3 0 0 1 20 9.5a3 3 0 0 1-1.5 5.5A3 3 0 0 1 12 18Z" />
    <path d="M12 5v13" />
  </Svg>
)
export const IconSettings = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2V21a2 2 0 1 1-4 0v-.1A1.7 1.7 0 0 0 7 19.4a1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0-1.2-2.9H1a2 2 0 1 1 0-4h.1A1.7 1.7 0 0 0 2.6 7a1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.7 1.7 0 0 0 8 1.6h.1A2 2 0 1 1 12 1.6V2a1.7 1.7 0 0 0 2.9 1.2l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0 1.2 2.9H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" />
  </Svg>
)
export const IconSun = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Svg>
)
export const IconMoon = (p: IconProps) => (
  <Svg {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" /></Svg>
)
export const IconMonitor = (p: IconProps) => (
  <Svg {...p}><rect x="2" y="4" width="20" height="13" rx="2" /><path d="M8 21h8M12 17v4" /></Svg>
)
export const IconSend = (p: IconProps) => (
  <Svg {...p}><path d="M4 12 20 4l-7 16-2.4-6.6L4 12Z" /></Svg>
)
export const IconStop = (p: IconProps) => (
  <Svg {...p}><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" /></Svg>
)
export const IconPlus = (p: IconProps) => (
  <Svg {...p}><path d="M12 5v14M5 12h14" /></Svg>
)
export const IconPaperclip = (p: IconProps) => (
  <Svg {...p}><path d="M21 11.5 12.5 20a5 5 0 0 1-7-7l8.5-8.5a3.5 3.5 0 0 1 5 5L10.4 18a2 2 0 0 1-3-3l8-8" /></Svg>
)
export const IconFolderPlus = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />
    <path d="M12 11v5M9.5 13.5h5" />
  </Svg>
)
export const IconChevronRight = (p: IconProps) => (
  <Svg {...p}><path d="m9 5 7 7-7 7" /></Svg>
)
export const IconChevronDown = (p: IconProps) => (
  <Svg {...p}><path d="m5 9 7 7 7-7" /></Svg>
)
export const IconClose = (p: IconProps) => (
  <Svg {...p}><path d="M18 6 6 18M6 6l12 12" /></Svg>
)
export const IconRefresh = (p: IconProps) => (
  <Svg {...p}><path d="M20 11a8 8 0 1 0-.7 4.4M20 5v6h-6" /></Svg>
)
export const IconCheck = (p: IconProps) => (
  <Svg {...p}><path d="m5 13 4 4L19 7" /></Svg>
)
export const IconAlert = (p: IconProps) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M12 7v6M12 16.5v.5" /></Svg>
)
export const IconSearch = (p: IconProps) => (
  <Svg {...p}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></Svg>
)
export const IconBolt = (p: IconProps) => (
  <Svg {...p}><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" /></Svg>
)
export const IconRestore = (p: IconProps) => (
  <Svg {...p}><path d="M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5" /></Svg>
)
export const IconBranch = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="7" cy="5" r="2.2" /><circle cx="7" cy="19" r="2.2" /><circle cx="17" cy="9" r="2.2" />
    <path d="M7 7.2v9.6M17 11.2c0 3-3 3.6-5.4 4.3" />
  </Svg>
)
export const IconKey = (p: IconProps) => (
  <Svg {...p}><circle cx="8" cy="14" r="4" /><path d="m11 11 8-8 2 2-2 2 2 2-2 2-2-2-2 2" /></Svg>
)
export const IconCpu = (p: IconProps) => (
  <Svg {...p}>
    <rect x="6" y="6" width="12" height="12" rx="1.5" /><rect x="9.5" y="9.5" width="5" height="5" rx="0.5" />
    <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
  </Svg>
)
export const IconTrash = (p: IconProps) => (
  <Svg {...p}><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13h10l1-13" /></Svg>
)
