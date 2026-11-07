import { useRef, useMemo, useEffect } from "react";
import type { RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/**
 * Configuration options for the gradient orb.
 * All fields are optional and fall back to sensible defaults.
 */
export type GradientOrbConfig = {
  /** CSS color string for the canvas background, or "transparent". @default "#0a0a0a" */
  background?: string;
  /** Hue rotation in degrees applied to all gradient colors. @default 0 */
  hue?: number;
  /** Constant rotation speed of the orb (radians/sec). @default 0.3 */
  rotationSpeed?: number;
  /** Scale of the noise pattern inside the orb. @default 0.65 */
  noiseScale?: number;
  /** Inner radius of the orb glow (0–1). @default 0.1 */
  innerRadius?: number;
  /** The three gradient colors as CSS strings. @default blue, purple, orange */
  colors?: [string, string, string];
  /** Lean, highlight, hue-shift and spin up toward the pointer. @default true */
  interactive?: boolean;
  /** Freeze time-based motion; pointer reaction still works. @default false */
  still?: boolean;
};

const defaults: Required<GradientOrbConfig> = {
  background: "#0a0a0a",
  hue: 0,
  rotationSpeed: 0.3,
  noiseScale: 0.65,
  innerRadius: 0.1,
  colors: ["#3d5aff", "#9d00ff", "#ff5f1f"],
  interactive: true,
  still: false,
};

/** How far the orb drifts toward the pointer, in orb radii. */
const LEAN = 0.12;
/** Degrees of hue shift when the pointer is at the orb's left/right edge. */
const HUE_SWING = 28;
/** Pointer distance (orb radii) beyond which the orb stops reacting. */
const REACH = 2.6;
/** Per-second smoothing rate for pointer-driven values. */
const EASE = 6;

/**
 * GLSL vertex shader — pass-through that outputs clip-space positions
 * directly from a fullscreen triangle in NDC.
 */
const vertexShader = /* glsl */ `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/**
 * GLSL fragment shader — renders a glowing orb with three noise-mixed colors,
 * hue rotation, breathing pulse, constant rotation and a pointer-facing rim light.
 */
const fragmentShader = /* glsl */ `
  precision highp float;

  uniform float iTime;
  uniform vec3 iResolution;
  uniform float hue;
  uniform float rot;
  uniform float noiseScale;
  uniform float innerRadius;
  uniform vec3 color0In;
  uniform vec3 color1In;
  uniform vec3 color2In;
  uniform vec2 mouse;
  uniform float hover;

  varying vec2 vUv;

  // --- YIQ color space hue rotation ---

  vec3 rgb2yiq(vec3 c) {
    return vec3(
      dot(c, vec3(0.299, 0.587, 0.114)),
      dot(c, vec3(0.596, -0.274, -0.322)),
      dot(c, vec3(0.211, -0.523, 0.312))
    );
  }

  vec3 yiq2rgb(vec3 c) {
    return vec3(
      c.x + 0.956 * c.y + 0.621 * c.z,
      c.x - 0.272 * c.y - 0.647 * c.z,
      c.x - 1.106 * c.y + 1.703 * c.z
    );
  }

  vec3 adjustHue(vec3 color, float hueDeg) {
    float hueRad = radians(hueDeg);
    vec3 yiq = rgb2yiq(color);
    float cosA = cos(hueRad);
    float sinA = sin(hueRad);
    yiq.yz = vec2(yiq.y * cosA - yiq.z * sinA, yiq.y * sinA + yiq.z * cosA);
    return yiq2rgb(yiq);
  }

  // --- 3D simplex noise (hash-based) ---

  vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(0.1031, 0.11369, 0.13787));
    p3 += dot(p3, p3.yxz + 19.19);
    return -1.0 + 2.0 * fract(vec3(p3.x + p3.y, p3.x + p3.z, p3.y + p3.z) * p3.zyx);
  }

  float snoise3(vec3 p) {
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;
    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
    vec3 i1 = e * (1.0 - e.zxy);
    vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    vec3 d1 = d0 - (i1 - K2);
    vec3 d2 = d0 - (i2 - K1);
    vec3 d3 = d0 - 0.5;
    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(
      dot(d0, hash33(i)),
      dot(d1, hash33(i + i1)),
      dot(d2, hash33(i + i2)),
      dot(d3, hash33(i + 1.0))
    );
    return dot(vec4(31.316), n);
  }

  // --- Orb rendering ---

  vec4 extractAlpha(vec3 colorIn) {
    float a = max(max(colorIn.r, colorIn.g), colorIn.b);
    return vec4(colorIn.rgb / (a + 1e-5), a);
  }

  const vec3 baseColor3 = vec3(0.0, 0.0, 0.0);       // black

  float light1(float intensity, float attenuation, float dist) {
    return intensity / (1.0 + dist * attenuation);
  }

  float light2(float intensity, float attenuation, float dist) {
    return intensity / (1.0 + dist * dist * attenuation);
  }

  vec4 draw(vec2 uv, vec2 m) {
    vec3 color0 = adjustHue(color0In, hue);
    vec3 color1 = adjustHue(color1In, hue);
    vec3 color2 = adjustHue(color2In, hue);
    vec3 color3 = adjustHue(baseColor3, hue);

    float len = length(uv);
    float invLen = len > 0.0 ? 1.0 / len : 0.0;

    float pulse = sin(iTime * 1.5) * 0.02 + hover * 0.03;

    float n0 = snoise3(vec3(uv * noiseScale, iTime * 0.5)) * 0.5 + 0.5;

    float r0 = mix(mix(innerRadius + pulse, 1.0, 0.4), mix(innerRadius + pulse, 1.0, 0.6), n0);

    float d0 = distance(uv, (r0 * invLen) * uv);
    float v0 = light1(1.0, 10.0, d0);
    v0 *= smoothstep(r0 * 1.05, r0, len);
    float cl = cos(atan(uv.y, uv.x) + iTime * 2.0) * 0.5 + 0.5;

    float a = iTime * -1.0;
    vec2 pos = vec2(cos(a), sin(a)) * r0;
    float d = distance(uv, pos);
    float v1 = light2(1.5, 5.0, d);

    // Second rim light on the side facing the pointer.
    float mLen = length(m);
    vec2 mPos = (mLen > 1e-4 ? m / mLen : vec2(0.0)) * r0;
    v1 += light2(1.6 * hover, 6.0, distance(uv, mPos));

    v1 *= light1(1.0, 50.0, d0);

    float v2 = smoothstep(1.0, mix(innerRadius, 1.0, n0 * 0.5), len);
    float v3 = smoothstep(innerRadius, mix(innerRadius, 1.0, 0.5), len);

    vec3 col = mix(color1, color2, cl);
    col = mix(col, color0, n0);
    col = mix(color3, col, v0);
    col = (col + v1) * v2 * v3;
    col = clamp(col, 0.0, 1.0);

    return extractAlpha(col);
  }

  vec2 rotate(vec2 p, float r) {
    float s = sin(r);
    float c = cos(r);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
  }

  void main() {
    vec2 center = iResolution.xy * 0.5;
    float size = min(iResolution.x, iResolution.y);
    vec2 uv = (vUv * iResolution.xy - center) / size * 2.0;

    // Lean toward the pointer, then spin.
    vec2 lean = mouse * ${LEAN.toFixed(3)} * hover;
    uv -= lean;

    vec4 col = draw(rotate(uv, rot), rotate(mouse - lean, rot));
    gl_FragColor = vec4(col.rgb * col.a, col.a);
  }
`;

/** CSS color -> sRGB components, written straight into a shader uniform. */
function setSrgb(target: THREE.Vector3, css: string) {
  const rgb = new THREE.Color(css).getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace);
  target.set(rgb.r, rgb.g, rgb.b);
}

/**
 * Tracks the pointer anywhere on the page, in orb space: the orb's centre is
 * (0,0) and its short side spans -1..1. Stored in a ref so it never re-renders.
 */
function usePointer(container: RefObject<HTMLDivElement | null>, enabled: boolean) {
  const pointer = useRef({ x: 0, y: 0, inside: false });

  useEffect(() => {
    if (!enabled) return;
    const onMove = (e: PointerEvent) => {
      const el = container.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const half = Math.min(r.width, r.height) / 2 || 1;
      pointer.current = {
        x: (e.clientX - (r.left + r.width / 2)) / half,
        y: -(e.clientY - (r.top + r.height / 2)) / half,
        inside: true,
      };
    };
    const onLeave = () => {
      pointer.current = { ...pointer.current, inside: false };
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    document.documentElement.addEventListener("pointerleave", onLeave);
    window.addEventListener("blur", onLeave);
    return () => {
      window.removeEventListener("pointermove", onMove);
      document.documentElement.removeEventListener("pointerleave", onLeave);
      window.removeEventListener("blur", onLeave);
    };
  }, [container, enabled]);

  return pointer;
}

type Pointer = ReturnType<typeof usePointer>;

/** Internal scene component for the gradient fullscreen shader. */
function GradientScene({ config, pointer }: { config: Required<GradientOrbConfig>; pointer: Pointer }) {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const { size, viewport } = useThree();
  const timeRef = useRef(0);
  const rotRef = useRef(0);
  const smooth = useRef({ x: 0, y: 0, hover: 0 });

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3),
    );
    geo.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
    return geo;
  }, []);

  useEffect(() => {
    return () => geometry.dispose();
  }, [geometry]);

  // Created once; every value is refreshed in useFrame, so config changes
  // (including colors) apply on the next frame without rebuilding the material.
  const uniforms = useMemo(
    () => ({
      iTime: { value: 0 },
      iResolution: { value: new THREE.Vector3(1, 1, 1) },
      hue: { value: 0 },
      rot: { value: 0 },
      noiseScale: { value: 0 },
      innerRadius: { value: 0 },
      color0In: { value: new THREE.Vector3() },
      color1In: { value: new THREE.Vector3() },
      color2In: { value: new THREE.Vector3() },
      mouse: { value: new THREE.Vector2() },
      hover: { value: 0 },
    }),
    [],
  );

  const [c0, c1, c2] = config.colors;
  useEffect(() => {
    setSrgb(uniforms.color0In.value, c0);
    setSrgb(uniforms.color1In.value, c1);
    setSrgb(uniforms.color2In.value, c2);
  }, [uniforms, c0, c1, c2]);

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.1); // a backgrounded tab returns one huge delta
    const p = pointer.current;
    const s = smooth.current;
    const live = config.interactive && p.inside;
    const reach = live ? Math.max(0, 1 - Math.hypot(p.x, p.y) / REACH) : 0;
    const k = 1 - Math.exp(-EASE * dt);
    s.x += ((live ? p.x : 0) - s.x) * k;
    s.y += ((live ? p.y : 0) - s.y) * k;
    s.hover += (reach - s.hover) * k;

    if (!config.still) timeRef.current += dt;
    rotRef.current += dt * config.rotationSpeed * (config.still ? 0 : 1 + s.hover * 2);

    // Write through the material (R3F's per-frame mutation pattern), not the memoised object.
    const u = materialRef.current?.uniforms as typeof uniforms | undefined;
    if (!u) return;
    u.iTime.value = timeRef.current;
    u.hue.value = config.hue + Math.max(-1, Math.min(1, s.x)) * HUE_SWING * s.hover;
    u.rot.value = rotRef.current;
    u.noiseScale.value = config.noiseScale;
    u.innerRadius.value = config.innerRadius;
    u.mouse.value.set(s.x, s.y);
    u.hover.value = s.hover;
    u.iResolution.value.set(
      size.width * viewport.dpr,
      size.height * viewport.dpr,
      size.width / size.height,
    );
  });

  return (
    <mesh geometry={geometry} frustumCulled={false}>
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
        premultipliedAlpha
        depthWrite={false}
        depthTest={false}
      />
    </mesh>
  );
}

/**
 * Glowing shader orb with noise-based 3-color mixing, YIQ hue rotation,
 * breathing pulse, constant rotation and pointer reaction — all rendered on
 * GPU as a single fullscreen triangle.
 *
 * @example
 * ```tsx
 * <GradientOrb />
 * <GradientOrb config={{ hue: 120, rotationSpeed: 0.5 }} />
 * <GradientOrb config={{ background: "transparent", colors: ["#e0713a", "#b0233f", "#ffd27a"] }} />
 * ```
 */
export function GradientOrb({
  config: configOverrides,
  className = "",
}: {
  config?: GradientOrbConfig;
  className?: string;
}) {
  const configKey = JSON.stringify(configOverrides);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const config = useMemo(() => ({ ...defaults, ...configOverrides }), [configKey]);
  const container = useRef<HTMLDivElement>(null);
  const pointer = usePointer(container, config.interactive);
  const transparent = config.background === "transparent";

  return (
    <div
      ref={container}
      className={className}
      style={{ width: "100%", height: "100%", background: transparent ? undefined : config.background }}
    >
      {/* Camera is unused — the vertex shader outputs clip-space positions directly */}
      <Canvas gl={{ antialias: true, alpha: transparent }} dpr={[1, 2]}>
        {!transparent && <color attach="background" args={[config.background]} />}
        <GradientScene config={config} pointer={pointer} />
      </Canvas>
    </div>
  );
}

export default GradientOrb;
