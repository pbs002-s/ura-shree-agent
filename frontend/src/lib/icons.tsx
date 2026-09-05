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
export const IconEdit = (p: IconProps) => (
  <Svg {...p}><path d="M12 20h9M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" /></Svg>
)
export const IconFolder = (p: IconProps) => (
  <Svg {...p}><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></Svg>
)

export const IconCopy = (p: IconProps) => (
  <Svg {...p}>
    <rect x="9" y="9" width="12" height="12" rx="2" />
    <path d="M6 15H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v1" />
  </Svg>
)
export const IconDownload = (p: IconProps) => (
  <Svg {...p}><path d="M12 3v12M7.5 10.5 12 15l4.5-4.5M4 20h16" /></Svg>
)
export const IconExternal = (p: IconProps) => (
  <Svg {...p}><path d="M14 4h6v6M20 4l-8.5 8.5M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" /></Svg>
)
export const IconCommand = (p: IconProps) => (
  <Svg {...p}>
    <path d="M9 6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3Z" />
  </Svg>
)
export const IconShield = (p: IconProps) => (
  <Svg {...p}><path d="M12 3l7 3v6c0 4.4-2.9 8.2-7 9-4.1-.8-7-4.6-7-9V6Z" /><path d="m9 12 2 2 4-4" /></Svg>
)
export const IconFlask = (p: IconProps) => (
  <Svg {...p}><path d="M9 3h6M10 3v6.2L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 9.2V3" /><path d="M7.5 15h9" /></Svg>
)
export const IconBug = (p: IconProps) => (
  <Svg {...p}>
    <rect x="8" y="7" width="8" height="13" rx="4" />
    <path d="M8 11H4M20 11h-4M8 16H4.5M20 16h-3.5M9.5 7 8 4.5M14.5 7 16 4.5" />
  </Svg>
)
export const IconBook = (p: IconProps) => (
  <Svg {...p}><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2Z" /><path d="M8 7h7M8 11h7" /></Svg>
)
export const IconCompass = (p: IconProps) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="m15.5 8.5-2 5-5 2 2-5Z" /></Svg>
)
export const IconGithub = (p: IconProps) => (
  <Svg {...p}>
    <path d="M9 19c-4.3 1.3-4.3-2.2-6-2.7m12 5.2v-3.4c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.3 4.3 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12 12 0 0 0-6.2 0C6.5 3.4 5.4 3.7 5.4 3.7a4.3 4.3 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 10.1c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21.5" />
  </Svg>
)
export const IconGlobe = (p: IconProps) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c2.5 2.6 2.5 15.4 0 18-2.5-2.6-2.5-15.4 0-18Z" /></Svg>
)
export const IconCamera = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3 8.5A1.5 1.5 0 0 1 4.5 7h2.3l1.4-2h7.6l1.4 2h2.3A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5Z" />
    <circle cx="12" cy="13" r="3.2" />
  </Svg>
)
export const IconPalette = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 3a9 9 0 0 0 0 18c1.4 0 2-1 2-2s-.7-2-.7-2.6c0-.7.5-1.4 1.5-1.4H18a3 3 0 0 0 3-3c0-4.4-4-8-9-8Z" />
    <circle cx="8" cy="10" r="1" /><circle cx="12" cy="7.5" r="1" /><circle cx="16" cy="10" r="1" />
  </Svg>
)
export const IconLayers = (p: IconProps) => (
  <Svg {...p}><path d="m12 3 9 5-9 5-9-5Z" /><path d="m4 13 8 4.5 8-4.5" /></Svg>
)
export const IconFileText = (p: IconProps) => (
  <Svg {...p}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" /><path d="M14 3v5h5M9 13h6M9 17h4" /></Svg>
)
export const IconImage = (p: IconProps) => (
  <Svg {...p}><rect x="3" y="5" width="18" height="14" rx="2" /><circle cx="9" cy="10" r="1.6" /><path d="m5 17 4.5-4.5L14 17l2.5-2.5L21 18" /></Svg>
)
export const IconCode = (p: IconProps) => (
  <Svg {...p}><path d="m8 8-4 4 4 4M16 8l4 4-4 4M13.5 6l-3 12" /></Svg>
)

export function IconUraShreeLogo({ size = 26, className = '' }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <defs>
        {/* The ramp is built from the accent token, so the mark follows the
            palette the user picked instead of a baked-in orange. */}
        <linearGradient id="uraShell" x1="4" y1="4" x2="44" y2="44" gradientUnits="userSpaceOnUse">
          <stop stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-hover)" />
        </linearGradient>
      </defs>
      <polygon points="24,3 43,14 43,34 24,45 5,34 5,14" fill="url(#uraShell)" />
      <polygon
        points="24,7.5 39,16 39,32 24,40.5 9,32 9,16"
        fill="none"
        stroke="var(--on-accent)"
        strokeOpacity="0.28"
        strokeWidth="1.2"
      />
      <path
        d="M31 16.5C29 14.8 26.5 14 23.5 14C18.8 14 16 16.8 16 20.2C16 23.8 19.2 25.2 23 26.2C27.5 27.4 30.5 28.8 30.5 32.8C30.5 37 26.8 39.5 22.2 39.5C18.2 39.5 15.2 37.8 13.5 35"
        stroke="var(--on-accent)"
        strokeWidth="3.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="31" cy="16.5" r="1.8" fill="var(--on-accent)" />
      <circle cx="13.5" cy="35" r="1.8" fill="var(--on-accent)" />
      <circle cx="23.2" cy="26.2" r="1.4" fill="var(--on-accent)" />
    </svg>
  )
}
