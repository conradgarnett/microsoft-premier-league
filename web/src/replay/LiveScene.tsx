import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { Team, Vec2, Vec3 } from "../api";

/** Latest two 2 fps frames from the live WebSocket feed; the scene interpolates between them. */
export interface LiveFeed {
  prev: { t: number; ball: Vec3; players: Vec2[]; at: number } | null;
  next: { t: number; ball: Vec3; players: Vec2[]; at: number } | null;
}

export default function LiveScene({ feed, teams, playerIds }: { feed: LiveFeed; teams: Team[]; playerIds: string[] }) {
  const bodies = useRef<THREE.InstancedMesh>(null!);
  const heads = useRef<THREE.InstancedMesh>(null!);
  const ball = useRef<THREE.Mesh>(null!);
  const cam = useRef(new THREE.Vector3(0, 0, 0));
  const colours = useMemo(() => {
    const byId = new Map(teams.flatMap((t) => t.players.map((p) => [p.id, p.position === "GK" ? "#c8e64a" : t.color_primary] as const)));
    return playerIds.map((id) => new THREE.Color(byId.get(id) ?? "#888"));
  }, [teams, playerIds]);
  const m = useMemo(() => new THREE.Matrix4(), []);
  const done = useRef(false);

  useFrame(({ camera }, dt) => {
    const { prev, next } = feed;
    if (!next) return;
    if (!done.current && bodies.current) {
      colours.forEach((c, i) => bodies.current.setColorAt(i, c));
      bodies.current.instanceColor!.needsUpdate = true;
      done.current = true;
    }
    // interpolate over the real-time gap between the last two messages
    const a = prev ? Math.min(1, (performance.now() - next.at) / Math.max(next.at - prev.at, 1)) : 1;
    const P = (i: number) => {
      const q = next.players[i], p = prev?.players[i] ?? q;
      return [p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a];
    };
    for (let i = 0; i < next.players.length; i++) {
      const [x, y] = P(i);
      m.makeTranslation(x - 52.5, 0.82, -(y - 34));
      bodies.current.setMatrixAt(i, m);
      m.makeScale(0.14, 0.16, 0.14).setPosition(x - 52.5, 1.78, -(y - 34));
      heads.current.setMatrixAt(i, m);
    }
    bodies.current.instanceMatrix.needsUpdate = true;
    heads.current.instanceMatrix.needsUpdate = true;
    const pb = prev?.ball ?? next.ball;
    const bx = pb[0] + (next.ball[0] - pb[0]) * a, by = pb[1] + (next.ball[1] - pb[1]) * a, bz = pb[2] + (next.ball[2] - pb[2]) * a;
    ball.current.position.set(bx - 52.5, Math.max(bz, 0.16), -(by - 34));
    // high broadcast camera tracking the ball
    const k = 1 - Math.exp(-dt * 1.5);
    cam.current.lerp(new THREE.Vector3((bx - 52.5) * 0.8, 0, -(by - 34) * 0.4), k);
    camera.position.set(cam.current.x, 34, 52);
    camera.lookAt(cam.current.x, 0, cam.current.z - 4);
  });

  return (
    <>
      <hemisphereLight args={["#bcd6ff", "#0c1a10", 0.6]} />
      <directionalLight position={[-30, 60, 25]} intensity={2.2} castShadow shadow-mapSize={[2048, 2048]} shadow-camera-left={-70} shadow-camera-right={70} shadow-camera-top={50} shadow-camera-bottom={-50} />
      <fog attach="fog" args={["#05070b", 90, 220]} />
      <instancedMesh ref={bodies} args={[undefined, undefined, playerIds.length]} castShadow frustumCulled={false}>
        <capsuleGeometry args={[0.24, 1.15, 6, 12]} />
        <meshStandardMaterial roughness={0.5} />
      </instancedMesh>
      <instancedMesh ref={heads} args={[undefined, undefined, playerIds.length]} castShadow frustumCulled={false}>
        <sphereGeometry args={[1, 14, 12]} />
        <meshStandardMaterial color="#c9a07e" roughness={0.6} />
      </instancedMesh>
      <mesh ref={ball} castShadow>
        <sphereGeometry args={[0.22, 20, 16]} />
        <meshStandardMaterial color="#fff" emissive="#fff" emissiveIntensity={0.4} />
      </mesh>
    </>
  );
}
