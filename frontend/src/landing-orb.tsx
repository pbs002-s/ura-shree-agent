// Mounts the hero orb on the static landing page (landing.html).
// The orb takes its colours from the page's live --accent and theme, and
// re-tints whenever the accent swatches or theme toggle change them.
import { useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import { GradientOrb } from "@/components/ui/gradient-orb";

const root = document.documentElement;

function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(root, { attributes: true, attributeFilter: ["data-accent", "data-theme"] });
  return () => observer.disconnect();
}

// A string snapshot keeps useSyncExternalStore's identity check cheap and stable.
const readThemeKey = () => `${root.dataset.theme ?? "dark"}|${root.dataset.accent ?? "teal"}`;

/** Accent in the middle, flanked by a cooler and a warmer neighbour on the colour wheel. */
function paletteFrom(accentCss: string, isDark: boolean): [string, string, string] {
  const accent = new THREE.Color(accentCss || "#0f766e");
  const lift = isDark ? 0.08 : -0.04;
  const cool = accent.clone().offsetHSL(-0.09, 0.1, lift);
  const warm = accent.clone().offsetHSL(0.07, 0.15, lift + 0.06);
  return [`#${accent.getHexString()}`, `#${cool.getHexString()}`, `#${warm.getHexString()}`];
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function LandingOrb() {
  const themeKey = useSyncExternalStore(subscribe, readThemeKey);
  const isDark = themeKey.startsWith("dark");
  const accent = getComputedStyle(root).getPropertyValue("--accent").trim();

  return (
    <GradientOrb
      config={{
        background: "transparent",
        colors: paletteFrom(accent, isDark),
        innerRadius: isDark ? 0.1 : 0.16,
        still: reducedMotion,
      }}
    />
  );
}

const mount = document.getElementById("hero-orb");
if (mount) createRoot(mount).render(<LandingOrb />);
