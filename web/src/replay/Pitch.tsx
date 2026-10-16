import { useMemo } from "react";
import * as THREE from "three";

// Pitch is 105 x 68 m, centred at the origin. World X = x - 52.5, Z = -(y - 34), Y = up.
const PX = 16; // texture pixels per metre
const MARGIN = 6;

function drawPitch(): HTMLCanvasElement {
  const w = (105 + MARGIN * 2) * PX;
  const h = (68 + MARGIN * 2) * PX;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d")!;
  g.fillStyle = "#1d4d2c";
  g.fillRect(0, 0, w, h);
  // mowing stripes
  for (let i = 0; i < 14; i++) {
    g.fillStyle = i % 2 ? "#2a6a3b" : "#2f7543";
    g.fillRect((MARGIN + i * 7.5) * PX, MARGIN * PX, 7.5 * PX, 68 * PX);
  }
  const m = (x: number) => (x + MARGIN) * PX;
  const my = (y: number) => (68 - y + MARGIN) * PX;
  g.strokeStyle = "rgba(255,255,255,0.92)";
  g.lineWidth = 0.12 * PX;
  g.strokeRect(m(0), my(68), 105 * PX, 68 * PX);
  g.beginPath();
  g.moveTo(m(52.5), my(0));
  g.lineTo(m(52.5), my(68));
  g.stroke();
  g.beginPath();
  g.arc(m(52.5), my(34), 9.15 * PX, 0, Math.PI * 2);
  g.stroke();
  for (const side of [0, 1]) {
    const x0 = side ? 105 - 16.5 : 0;
    g.strokeRect(m(x0), my(34 + 20.16), 16.5 * PX, 40.32 * PX);
    const x1 = side ? 105 - 5.5 : 0;
    g.strokeRect(m(x1), my(34 + 9.16), 5.5 * PX, 18.32 * PX);
    const spot = side ? 105 - 11 : 11;
    g.beginPath();
    g.arc(m(spot), my(34), 0.25 * PX, 0, Math.PI * 2);
    g.fillStyle = "white";
    g.fill();
    g.beginPath();
    const a = Math.acos(5.5 / 9.15);
    if (side) g.arc(m(spot), my(34), 9.15 * PX, Math.PI - a, Math.PI + a);
    else g.arc(m(spot), my(34), 9.15 * PX, -a, a);
    g.stroke();
  }
  g.beginPath();
  g.arc(m(52.5), my(34), 0.25 * PX, 0, Math.PI * 2);
  g.fill();
  return c;
}

function Goal({ side }: { side: 1 | -1 }) {
  const x = side * 52.5;
  const depth = 2;
  const mat = <meshStandardMaterial color="#f4f4f4" metalness={0.2} roughness={0.4} />;
  const net = useMemo(() => {
    const geo = new THREE.BoxGeometry(depth, 2.44, 7.32, 4, 6, 16);
    return new THREE.WireframeGeometry(geo);
  }, []);
  return (
    <group position={[x + side * depth / 2, 0, 0]}>
      {[-3.66, 3.66].map((z) => (
        <mesh key={z} position={[-side * depth / 2, 1.22, z]} castShadow>
          <cylinderGeometry args={[0.06, 0.06, 2.44, 12]} />
          {mat}
        </mesh>
      ))}
      <mesh position={[-side * depth / 2, 2.44, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
        <cylinderGeometry args={[0.06, 0.06, 7.32, 12]} />
        {mat}
      </mesh>
      <lineSegments geometry={net} position={[0, 1.22, 0]}>
        <lineBasicMaterial color="#d8dde3" transparent opacity={0.35} />
      </lineSegments>
    </group>
  );
}

function crowdTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 1024;
  c.height = 256;
  const g = c.getContext("2d")!;
  g.fillStyle = "#0a0d14";
  g.fillRect(0, 0, 1024, 256);
  // seat rows + a speckled crowd, slightly team-tinted
  for (let row = 0; row < 16; row++) {
    g.fillStyle = row % 2 ? "#121826" : "#0e131e";
    g.fillRect(0, row * 16, 1024, 16);
  }
  const tints = ["#2f81f7", "#f2683c", "#d9dde6", "#3a4152", "#6b7385"];
  for (let i = 0; i < 9000; i++) {
    g.fillStyle = tints[Math.floor(Math.random() * tints.length)];
    g.globalAlpha = 0.25 + Math.random() * 0.45;
    g.fillRect(Math.random() * 1024, Math.floor(Math.random() * 16) * 16 + 4 + Math.random() * 6, 3, 4);
  }
  g.globalAlpha = 1;
  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function boardTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 1024;
  c.height = 32;
  const g = c.getContext("2d")!;
  const grad = g.createLinearGradient(0, 0, 1024, 0);
  grad.addColorStop(0, "#0b2a4a");
  grad.addColorStop(0.5, "#123d6b");
  grad.addColorStop(1, "#0b2a4a");
  g.fillStyle = grad;
  g.fillRect(0, 0, 1024, 32);
  g.font = "bold 22px Barlow Condensed, Arial Narrow, sans-serif";
  g.fillStyle = "#9fd2ff";
  const text = "TACTICAL REPLAY  ·  INSIDE THE GAME";
  const step = g.measureText(text).width + 60; // fallback fonts are wider: space by measured width
  for (let x = 20; x + step - 60 < 1024; x += step) g.fillText(text, x, 24);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Ad boards on all sides; tiered stands behind the far touchline and both goals (the near side hosts the broadcast camera). */
function Stadium() {
  const crowd = useMemo(crowdTexture, []);
  const boards = useMemo(boardTexture, []);
  const sides: { pos: [number, number, number]; rot: number; len: number; stand: boolean }[] = [
    { pos: [0, 0, -38], rot: 0, len: 113, stand: true },
    { pos: [0, 0, 38], rot: Math.PI, len: 113, stand: false },
    { pos: [-56.5, 0, 0], rot: Math.PI / 2, len: 76, stand: true },
    { pos: [56.5, 0, 0], rot: -Math.PI / 2, len: 76, stand: true },
  ];
  const tilt = 0.95;
  const depth = 30;
  return (
    <>
      {sides.map((sd, i) => {
        const tex = boards.clone();
        tex.repeat.set(sd.len / 40, 1);
        tex.needsUpdate = true;
        const ctex = crowd.clone();
        ctex.repeat.set(sd.len / 30, 1);
        ctex.needsUpdate = true;
        return (
          <group key={i} position={sd.pos} rotation={[0, sd.rot, 0]}>
            <mesh position={[0, 0.5, 0]}>
              <boxGeometry args={[sd.len, 1, 0.2]} />
              <meshStandardMaterial map={tex} emissive="#3b8fe0" emissiveMap={tex} emissiveIntensity={0.8} />
            </mesh>
            {sd.stand && (
              <mesh position={[0, 2 + (depth / 2) * Math.cos(tilt), -4 - (depth / 2) * Math.sin(tilt)]} rotation={[-tilt, 0, 0]}>
                <planeGeometry args={[sd.len + 20, depth]} />
                <meshStandardMaterial map={ctex} roughness={1} />
              </mesh>
            )}
          </group>
        );
      })}
      {[[-62, -44], [62, -44], [-62, 44], [62, 44]].map(([x, z], i) => (
        <group key={i} position={[x, 0, z]}>
          <mesh position={[0, 18, 0]}>
            <cylinderGeometry args={[0.35, 0.5, 36, 8]} />
            <meshStandardMaterial color="#1a1f2b" />
          </mesh>
          <mesh position={[0, 36.5, 0]} rotation={[0, Math.atan2(-x, -z), 0]}>
            <boxGeometry args={[6, 2.4, 0.4]} />
            <meshStandardMaterial color="#fff" emissive="#fff8e0" emissiveIntensity={3} />
          </mesh>
        </group>
      ))}
    </>
  );
}

export default function Pitch() {
  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(drawPitch());
    t.anisotropy = 8;
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, []);
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[105 + MARGIN * 2, 68 + MARGIN * 2]} />
        <meshStandardMaterial map={texture} roughness={0.95} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
        <planeGeometry args={[400, 400]} />
        <meshStandardMaterial color="#06080c" />
      </mesh>
      <Goal side={1} />
      <Goal side={-1} />
      <Stadium />
    </group>
  );
}
