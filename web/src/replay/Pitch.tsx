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

function Stands() {
  // Simple dark tiers around the pitch with a faint crowd speckle; purely atmospheric.
  const tex = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = 512;
    c.height = 64;
    const g = c.getContext("2d")!;
    g.fillStyle = "#0b0f17";
    g.fillRect(0, 0, 512, 64);
    for (let i = 0; i < 2600; i++) {
      const v = 30 + Math.random() * 60;
      g.fillStyle = `rgb(${v},${v + 4},${v + 14})`;
      g.fillRect(Math.random() * 512, Math.random() * 64, 2, 2);
    }
    const t = new THREE.CanvasTexture(c);
    t.wrapS = THREE.RepeatWrapping;
    t.repeat.set(6, 1);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, []);
  const sides: [number, number, number, number][] = [
    [0, 44, 130, 0],
    [0, -44, 130, Math.PI],
    [64, 0, 90, -Math.PI / 2],
    [-64, 0, 90, Math.PI / 2],
  ];
  return (
    <>
      {sides.map(([x, z, len, rot], i) => (
        <mesh key={i} position={[x, 6, z]} rotation={[-0.55, rot, 0, "YXZ"]}>
          <planeGeometry args={[len, 16]} />
          <meshStandardMaterial map={tex} side={THREE.DoubleSide} roughness={1} />
        </mesh>
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
      <Stands />
    </group>
  );
}
