import React, { useEffect, useRef } from 'react';

export interface AsciiWavesProps {
  characters?: string;
  cellWidth?: number;
  cellHeight?: number;
  speed?: number;
  frequency?: number;
  enableMouseRipple?: boolean;
  rippleRadius?: number;
  opacity?: number;
  className?: string;
  style?: React.CSSProperties;
}

function parseHexColor(hex: string): { r: number; g: number; b: number } {
  if (!hex) return { r: 194, g: 85, b: 31 };
  const cleanHex = hex.replace('#', '').trim();
  const fullHex =
    cleanHex.length === 3
      ? cleanHex[0] + cleanHex[0] + cleanHex[1] + cleanHex[1] + cleanHex[2] + cleanHex[2]
      : cleanHex;
  const val = parseInt(fullHex, 16);
  if (isNaN(val)) return { r: 194, g: 85, b: 31 };
  return {
    r: (val >> 16) & 255,
    g: (val >> 8) & 255,
    b: val & 255,
  };
}

export const AsciiWaves: React.FC<AsciiWavesProps> = ({
  characters = ' .:-+*=%@#',
  cellWidth = 16,
  cellHeight = 18,
  speed = 1.0,
  frequency = 1.0,
  enableMouseRipple = true,
  rippleRadius = 450,
  opacity = 1.0,
  className = '',
  style,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let W = 0;
    let H = 0;
    let visible = true;
    let raf = 0;

    const mouse = { x: -2000, y: -2000, targetX: -2000, targetY: -2000, strength: 0 };

    const handleMouseMove = (e: MouseEvent) => {
      if (!enableMouseRipple) return;
      const rect = canvas.getBoundingClientRect();
      mouse.targetX = e.clientX - rect.left;
      mouse.targetY = e.clientY - rect.top;
      mouse.strength = 1.0;
    };

    const handleMouseLeave = () => {
      mouse.targetX = -2000;
      mouse.targetY = -2000;
      mouse.strength = 0;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);

    const resize = () => {
      const parent = canvas.parentElement;
      const w = parent ? parent.clientWidth : window.innerWidth;
      const h = parent ? parent.clientHeight : window.innerHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = canvas.width = Math.max(2, Math.round(w * dpr));
      H = canvas.height = Math.max(2, Math.round(h * dpr));
    };

    window.addEventListener('resize', resize);
    resize();

    const charLen = characters.length;
    const t0 = performance.now();

    const getTokens = () => {
      const root = document.documentElement;
      const cs = getComputedStyle(root);
      const accent = cs.getPropertyValue('--accent').trim() || '#c2551f';
      const isDark = root.dataset.theme === 'dark';
      const rgb = parseHexColor(accent);
      return {
        accent,
        rgb,
        isDark,
      };
    };

    const render = (now: number) => {
      if (!visible) return;
      const t = (now - t0) * 0.0014 * speed;
      const w = W / dpr;
      const h = H / dpr;

      if (enableMouseRipple) {
        mouse.x += (mouse.targetX - mouse.x) * 0.08;
        mouse.y += (mouse.targetY - mouse.y) * 0.08;
        mouse.strength *= 0.97;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const tok = getTokens();

      ctx.font = '11.5px ui-monospace, "SF Mono", "Cascadia Mono", "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      const cols = Math.ceil(w / cellWidth);
      const rows = Math.ceil(h / cellHeight);

      for (let r = 0; r < rows; r++) {
        const cy = r * cellHeight + cellHeight * 0.5;

        for (let c = 0; c < cols; c++) {
          const cx = c * cellWidth + cellWidth * 0.5;

          const nx = cx * 0.0075 * frequency;
          const ny = cy * 0.009 * frequency;

          const w1 = Math.sin(nx * 1.5 + t * 1.6 + Math.cos(ny * 1.8 + t * 0.9));
          const w2 = Math.cos(ny * 1.4 - t * 1.2 + Math.sin(nx * 1.2 - t * 1.5));
          const waveVal = (w1 + w2) * 0.5;

          let mRipple = 0;
          if (enableMouseRipple && mouse.strength > 0.01) {
            const mDist = Math.hypot(cx - mouse.x, cy - mouse.y);
            if (mDist < rippleRadius) {
              const mFactor = (1 - mDist / rippleRadius) * mouse.strength;
              mRipple = Math.sin(mDist * 0.04 - t * 5.0) * mFactor * 1.5;
            }
          }

          let normalized = (waveVal + mRipple + 1.0) * 0.5;
          if (normalized < 0) normalized = 0;
          if (normalized > 1) normalized = 1;

          const charIndex = Math.floor(normalized * (charLen - 1));
          const char = characters[charIndex];

          if (!char || char === ' ') continue;

          let red = 0;
          let green = 0;
          let blue = 0;
          let charAlpha = 0;

          if (tok.isDark) {
            charAlpha = 0.12 + normalized * 0.28;
            if (normalized > 0.45) {
              const mix = (normalized - 0.45) / 0.55;
              red = Math.round(210 * (1 - mix * 0.45) + tok.rgb.r * (mix * 0.45));
              green = Math.round(225 * (1 - mix * 0.45) + tok.rgb.g * (mix * 0.45));
              blue = Math.round(245 * (1 - mix * 0.45) + tok.rgb.b * (mix * 0.45));
            } else {
              red = 210;
              green = 225;
              blue = 245;
            }
          } else {
            // Light Mode: clean, crisp, slate-charcoal with theme accent harmony
            charAlpha = 0.13 + normalized * 0.27;
            if (normalized > 0.45) {
              const mix = (normalized - 0.45) / 0.55;
              const tr = Math.round(tok.rgb.r * 0.72);
              const tg = Math.round(tok.rgb.g * 0.72);
              const tb = Math.round(tok.rgb.b * 0.72);
              red = Math.round(48 * (1 - mix * 0.42) + tr * (mix * 0.42));
              green = Math.round(62 * (1 - mix * 0.42) + tg * (mix * 0.42));
              blue = Math.round(82 * (1 - mix * 0.42) + tb * (mix * 0.42));
            } else {
              red = 48;
              green = 62;
              blue = 82;
            }
          }

          ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${charAlpha})`;

          const edgeFadeY = Math.sin(Math.min(1, Math.max(0, cy / h)) * Math.PI);
          ctx.globalAlpha = (0.45 + 0.55 * edgeFadeY) * opacity;

          ctx.fillText(char, cx, cy);
        }
      }

      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(render);
    };

    const handleVisibility = () => {
      if (document.hidden) {
        visible = false;
        cancelAnimationFrame(raf);
      } else {
        visible = true;
        raf = requestAnimationFrame(render);
      }
    };

    document.addEventListener('visibilitychange', handleVisibility);
    raf = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
      window.removeEventListener('resize', resize);
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, [characters, cellWidth, cellHeight, speed, frequency, enableMouseRipple, rippleRadius, opacity]);

  return (
    <canvas
      ref={canvasRef}
      className={`ascii-waves-canvas ${className}`}
      style={{
        position: 'fixed',
        inset: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 0,
        pointerEvents: 'none',
        opacity: 1,
        ...style,
      }}
      aria-hidden="true"
    />
  );
};

export default AsciiWaves;
