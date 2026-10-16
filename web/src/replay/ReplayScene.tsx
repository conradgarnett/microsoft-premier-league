import { Html, Line, OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { ReelItem, Replay, Team, Vec3 } from "../api";

export type CameraMode = "director" | "broadcast" | "orbit" | "player" | "behind";
export interface Overlays {
  passes: boolean; pressure: boolean; defLine: boolean; sprints: boolean; speeds: boolean; angles: boolean; tags: boolean;
}
/** Mutable playback clock shared between the scene (60 fps) and the UI (10 Hz). */
export interface Clock { frame: number; playing: boolean; rate: number; autoSlow: boolean }

export const toWorld = (x: number, y: number, z = 0) => new THREE.Vector3(x - 52.5, z, -(y - 34));

interface Ctx {
  replay: Replay;
  teams: Team[];
  clock: Clock;
  overlays: Overlays;
  camera: CameraMode;
  focusId: string | null;
  script?: ReelItem;
}

/** Interpolated joints for frame f (fractional). Returns [player][joint] world vectors. */
function sampleJoints(r: Replay, f: number, out: THREE.Vector3[][]) {
  const fr = r.skeleton.frames;
  const f0 = Math.max(0, Math.min(fr.length - 1, Math.floor(f)));
  const f1 = Math.min(fr.length - 1, f0 + 1);
  const a = Math.min(Math.max(f - f0, 0), 1);
  for (let p = 0; p < fr[f0].length; p++) {
    for (let j = 0; j < fr[f0][p].length; j++) {
      const A = fr[f0][p][j], B = fr[f1][p][j];
      out[p][j].set(A[0] + (B[0] - A[0]) * a - 52.5, A[2] + (B[2] - A[2]) * a, -(A[1] + (B[1] - A[1]) * a - 34));
    }
  }
}

function sampleBall(r: Replay, f: number): THREE.Vector3 {
  const b = r.tracking.ball;
  const f0 = Math.max(0, Math.min(b.length - 1, Math.floor(f)));
  const f1 = Math.min(b.length - 1, f0 + 1);
  const a = Math.min(Math.max(f - f0, 0), 1);
  const lerp = (i: number) => b[f0][i] + (b[f1][i] - b[f0][i]) * a;
  return toWorld(lerp(0), lerp(1), Math.max(lerp(2), 0.11));
}

const BONE_RADIUS: Record<string, number> = {
  "neck-pelvis": 0.16, "head-neck": 0.05, "neck-l_shoulder": 0.07, "neck-r_shoulder": 0.07,
  "pelvis-l_hip": 0.11, "pelvis-r_hip": 0.11, "l_hip-l_knee": 0.085, "r_hip-r_knee": 0.085,
  "l_knee-l_ankle": 0.065, "r_knee-r_ankle": 0.065, "l_ankle-l_toe": 0.045, "r_ankle-r_toe": 0.045,
};

function Players({ replay, teams, clock, overlays, focusId }: Ctx) {
  const sk = replay.skeleton;
  const P = sk.player_ids.length;
  const nJ = sk.joint_names.length;
  const bones = useMemo(() => sk.bones.map(([a, b]) => [sk.joint_names.indexOf(a), sk.joint_names.indexOf(b), BONE_RADIUS[`${a}-${b}`] ?? 0.05] as const), [sk]);
  const joints = useMemo(() => Array.from({ length: P }, () => Array.from({ length: nJ }, () => new THREE.Vector3())), [P, nJ]);
  const boneMesh = useRef<THREE.InstancedMesh>(null!);
  const headMesh = useRef<THREE.InstancedMesh>(null!);
  const jointMesh = useRef<THREE.InstancedMesh>(null!);
  const tagRefs = useRef<(THREE.Group | null)[]>([]);
  const speedRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const tagDivs = useRef<(HTMLDivElement | null)[]>([]);
  const { camera } = useThree();
  const ringRef = useRef<THREE.Mesh>(null!);
  const playerInfo = useMemo(() => {
    const byId = new Map(teams.flatMap((t) => t.players.map((p) => [p.id, { p, t }] as const)));
    return sk.player_ids.map((id) => byId.get(id)!);
  }, [teams, sk]);

  // instance colours: shirt (primary), shorts (secondary-dark), socks (primary), skin
  useMemo(() => {
    queueMicrotask(() => {
      if (!boneMesh.current) return;
      const c = new THREE.Color();
      playerInfo.forEach(({ p, t }, pi) => {
        const gk = p.position === "GK";
        const shirt = new THREE.Color(gk ? (t.id === teams[0].id ? "#c8e64a" : "#9b59ff") : t.color_primary);
        const shorts = new THREE.Color(t.color_secondary).lerp(new THREE.Color("#111"), 0.2);
        bones.forEach(([a, b], bi) => {
          const name = `${sk.joint_names[a]}-${sk.joint_names[b]}`;
          if (name.includes("hip-") || name.startsWith("pelvis")) c.copy(shorts);
          else if (name.includes("knee-") || name.includes("ankle")) c.copy(name.includes("toe") ? new THREE.Color("#222") : shirt);
          else if (name.includes("elbow-") || name.includes("wrist")) c.set("#c9a07e");
          else c.copy(shirt);
          boneMesh.current.setColorAt(pi * bones.length + bi, c);
        });
        headMesh.current.setColorAt(pi, new THREE.Color("#c9a07e"));
        for (let j = 0; j < nJ; j++) jointMesh.current.setColorAt(pi * nJ + j, j < 8 ? shirt : shorts);
      });
      boneMesh.current.instanceColor!.needsUpdate = true;
      headMesh.current.instanceColor!.needsUpdate = true;
      jointMesh.current.instanceColor!.needsUpdate = true;
    });
  }, [playerInfo, bones, sk, nJ, teams]);

  const tmp = useMemo(() => ({ m: new THREE.Matrix4(), q: new THREE.Quaternion(), s: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0), d: new THREE.Vector3(), mid: new THREE.Vector3() }), []);

  useFrame(() => {
    const f = clock.frame;
    sampleJoints(replay, f, joints);
    const { m, q, s, up, d, mid } = tmp;
    let k = 0;
    for (let p = 0; p < P; p++) {
      const J = joints[p];
      // a body brushing the lens fills the frame: drop it (never the followed player)
      const vis = sk.player_ids[p] !== focusId && camera.position.distanceTo(J[8]) < 2.5 ? 0 : 1;
      for (const [a, b, r] of bones) {
        d.subVectors(J[b], J[a]);
        const len = d.length();
        mid.addVectors(J[a], J[b]).multiplyScalar(0.5);
        q.setFromUnitVectors(up, d.normalize());
        s.set(r * vis, len * vis, r * vis);
        m.compose(mid, q, s);
        boneMesh.current.setMatrixAt(k++, m);
      }
      m.compose(J[0], q.identity(), s.set(0.13 * vis, 0.15 * vis, 0.13 * vis));
      headMesh.current.setMatrixAt(p, m);
      for (let j = 0; j < nJ; j++) {
        const rr = (j === 0 ? 0.001 : j === 8 ? 0.15 : j < 8 ? 0.06 : 0.07) * vis;
        m.compose(J[j], q.identity(), s.set(rr, rr, rr));
        jointMesh.current.setMatrixAt(p * nJ + j, m);
      }
      const g = tagRefs.current[p];
      if (g) g.position.set(J[0].x, J[0].y + 0.45, J[0].z);
      // close-up cameras: hide tags that would fill the screen
      const div = tagDivs.current[p];
      if (div) div.style.visibility = camera.position.distanceTo(J[0]) < 6 ? "hidden" : "visible";
    }
    boneMesh.current.instanceMatrix.needsUpdate = true;
    headMesh.current.instanceMatrix.needsUpdate = true;
    jointMesh.current.instanceMatrix.needsUpdate = true;

    // speeds (km/h) from tracking
    const tr = replay.tracking.players;
    const f0 = Math.max(1, Math.min(tr.length - 1, Math.round(f)));
    for (let p = 0; p < P; p++) {
      const el = speedRefs.current[p];
      if (!el) continue;
      const a = tr[Math.max(0, f0 - 3)][p], b = tr[f0][p];
      const v = (Math.hypot(b[0] - a[0], b[1] - a[1]) * replay.tracking.fps) / Math.min(3, f0) * 3.6;
      el.textContent = overlays.speeds ? ` ${v.toFixed(0)} km/h` : "";
      el.style.color = v > 25 ? "#ffb347" : "";
    }
    if (ringRef.current && focusId) {
      const p = sk.player_ids.indexOf(focusId);
      if (p >= 0) ringRef.current.position.set(joints[p][8].x, 0.03, joints[p][8].z);
    }
  });

  return (
    <group>
      <instancedMesh ref={boneMesh} args={[undefined, undefined, P * bones.length]} castShadow frustumCulled={false}>
        <cylinderGeometry args={[1, 1, 1, 10]} />
        <meshStandardMaterial roughness={0.55} />
      </instancedMesh>
      <instancedMesh ref={jointMesh} args={[undefined, undefined, P * nJ]} castShadow frustumCulled={false}>
        <sphereGeometry args={[1, 12, 10]} />
        <meshStandardMaterial roughness={0.55} />
      </instancedMesh>
      <instancedMesh ref={headMesh} args={[undefined, undefined, P]} castShadow frustumCulled={false}>
        <sphereGeometry args={[1, 18, 14]} />
        <meshStandardMaterial roughness={0.6} />
      </instancedMesh>
      {focusId && (
        <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.7, 0.9, 40]} />
          <meshBasicMaterial color="#ffe14d" transparent opacity={0.9} />
        </mesh>
      )}
      {overlays.tags &&
        playerInfo.map(({ p, t }, i) => (
          <group key={p.id} ref={(el) => (tagRefs.current[i] = el)}>
            <Html center zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
              <div ref={(el) => (tagDivs.current[i] = el)} className={`tag ${p.id === focusId ? "focus" : ""}`} style={{ borderColor: t.color_primary }}>
                <b>{p.number}</b> {p.short_name.split(" ").slice(-1)[0]}
                <span ref={(el) => (speedRefs.current[i] = el)} className="spd" />
              </div>
            </Html>
          </group>
        ))}
    </group>
  );
}

function Ball({ replay, clock }: Ctx) {
  const ref = useRef<THREE.Mesh>(null!);
  const shadow = useRef<THREE.Mesh>(null!);
  const trail = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(16 * 3), 3));
    return g;
  }, []);
  useFrame(() => {
    const p = sampleBall(replay, clock.frame);
    ref.current.position.copy(p);
    shadow.current.position.set(p.x, 0.02, p.z);
    const arr = trail.attributes.position.array as Float32Array;
    for (let i = 0; i < 16; i++) {
      const q = sampleBall(replay, clock.frame - i * 1.0);
      arr.set([q.x, q.y, q.z], i * 3);
    }
    trail.attributes.position.needsUpdate = true;
  });
  return (
    <group>
      <mesh ref={ref} castShadow>
        <sphereGeometry args={[0.16, 24, 18]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.25} roughness={0.3} />
      </mesh>
      <mesh ref={shadow} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.22, 20]} />
        <meshBasicMaterial color="#000" transparent opacity={0.35} />
      </mesh>
      <line>
        <primitive object={trail} attach="geometry" />
        <lineBasicMaterial color="#fff6c2" transparent opacity={0.7} />
      </line>
    </group>
  );
}

/** Index of the player closest to the ball, i.e. the likely ball carrier. */
function carrier(r: Replay, f: number): number {
  const fi = Math.max(0, Math.min(r.tracking.n_frames - 1, Math.round(f)));
  const b = r.tracking.ball[fi];
  let best = -1, bd = 1e9;
  r.tracking.players[fi].forEach((p, i) => {
    const d = Math.hypot(p[0] - b[0], p[1] - b[1]);
    if (d < bd) { bd = d; best = i; }
  });
  return bd < 2.5 && b[2] < 1.5 ? best : -1;
}

function TacticalOverlays({ replay, teams, clock, overlays }: Ctx) {
  const rings = useRef<THREE.InstancedMesh>(null!);
  const line = useRef<THREE.Mesh>(null!);
  const lineLabel = useRef<HTMLDivElement>(null);
  const trailGeos = useMemo(() => replay.tracking.player_ids.map(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(30 * 3), 3));
    return g;
  }), [replay]);
  const trailMats = useRef<(THREE.LineBasicMaterial | null)[]>([]);
  const teamOf = useMemo(() => {
    const m = new Map(teams.flatMap((t) => t.players.map((p) => [p.id, t.id] as const)));
    return replay.tracking.player_ids.map((id) => m.get(id)!);
  }, [teams, replay]);
  const gkIdx = useMemo(() => new Set(teams.flatMap((t) => t.players.filter((p) => p.position === "GK").map((p) => replay.tracking.player_ids.indexOf(p.id)))), [teams, replay]);
  const tmp = useMemo(() => new THREE.Matrix4(), []);

  useFrame(() => {
    const f = clock.frame;
    const fi = Math.max(0, Math.min(replay.tracking.n_frames - 1, Math.round(f)));
    const pos = replay.tracking.players[fi];
    const c = carrier(replay, f);
    // pressure: ring under each opponent within 6 m of the carrier, brighter when closer
    let k = 0;
    if (overlays.pressure && c >= 0) {
      const col = new THREE.Color();
      pos.forEach((p, i) => {
        if (teamOf[i] === teamOf[c]) return;
        const d = Math.hypot(p[0] - pos[c][0], p[1] - pos[c][1]);
        if (d > 6) return;
        const w = 1 - d / 6;
        const v = toWorld(p[0], p[1], 0.04);
        tmp.compose(v, new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)), new THREE.Vector3(1 + w, 1 + w, 1));
        rings.current.setMatrixAt(k, tmp);
        rings.current.setColorAt(k, col.setRGB(1, 0.25 + 0.5 * (1 - w), 0.15).multiplyScalar(0.4 + w));
        k++;
      });
      const cv = toWorld(pos[c][0], pos[c][1], 0.035);
      tmp.compose(cv, new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)), new THREE.Vector3(3.2, 3.2, 1));
      rings.current.setMatrixAt(k, tmp);
      rings.current.setColorAt(k, new THREE.Color("#ff5a36").multiplyScalar(0.35));
      k++;
    }
    rings.current.count = k;
    rings.current.instanceMatrix.needsUpdate = true;
    if (rings.current.instanceColor) rings.current.instanceColor.needsUpdate = true;

    // defensive line of the team without the ball: second-deepest player incl. keeper (offside line)
    line.current.visible = overlays.defLine && c >= 0;
    if (c >= 0) {
      const defTeam = teamOf.find((t) => t !== teamOf[c])!;
      const xs = pos.map((p, i) => (teamOf[i] === defTeam ? p[0] : NaN)).filter((x) => !Number.isNaN(x));
      // the defending keeper marks which goal they protect
      const gk = [...gkIdx].find((i) => teamOf[i] === defTeam);
      // no keeper known (video clips): defenders' mean position marks their goal side
      const goalRight = gk !== undefined ? pos[gk][0] > 52.5 : xs.reduce((a, b) => a + b, 0) / xs.length > pos[c][0];
      const sorted = [...xs].sort((a, b) => (goalRight ? b - a : a - b));
      const lx = sorted[1];
      line.current.position.set(lx - 52.5, 0.04, 0);
      if (lineLabel.current) {
        const dist = goalRight ? 105 - lx : lx;
        lineLabel.current.textContent = `Defensive line ${dist.toFixed(1)} m`;
      }
    }

    // sprint trails: last 1.2 s for anyone above 7 m/s
    replay.tracking.player_ids.forEach((_, i) => {
      const mat = trailMats.current[i];
      if (!mat) return;
      const a = replay.tracking.players[Math.max(0, fi - 5)][i];
      const v = (Math.hypot(pos[i][0] - a[0], pos[i][1] - a[1]) * replay.tracking.fps) / 5;
      mat.visible = overlays.sprints && v > 7 && !gkIdx.has(i);
      if (!mat.visible) return;
      const arr = trailGeos[i].attributes.position.array as Float32Array;
      for (let s = 0; s < 30; s++) {
        const q = replay.tracking.players[Math.max(0, fi - s)][i];
        arr.set([q[0] - 52.5, 0.06, -(q[1] - 34)], s * 3);
      }
      trailGeos[i].attributes.position.needsUpdate = true;
    });
  });

  return (
    <group>
      <instancedMesh ref={rings} args={[undefined, undefined, 24]} frustumCulled={false}>
        <ringGeometry args={[0.6, 0.85, 36]} />
        <meshBasicMaterial transparent opacity={0.85} blending={THREE.AdditiveBlending} depthWrite={false} />
      </instancedMesh>
      <mesh ref={line} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.35, 68]} />
        <meshBasicMaterial color="#5ce1ff" transparent opacity={0.75} />
        <Html position={[0, -31, 0.5]} center zIndexRange={[5, 0]} style={{ pointerEvents: "none" }}>
          <div ref={lineLabel} className="callout line-label" />
        </Html>
      </mesh>
      {trailGeos.map((g, i) => (
        <line key={i}>
          <primitive object={g} attach="geometry" />
          <lineBasicMaterial ref={(m: THREE.LineBasicMaterial | null) => (trailMats.current[i] = m)} color="#ffb347" transparent opacity={0.9} />
        </line>
      ))}
    </group>
  );
}

const diffColor = (d: number) => new THREE.Color().setHSL(0.33 * (1 - Math.min(Math.max(d, 0), 1)), 0.9, 0.55);

/** Event-driven overlays (pass lines, strike callouts) re-rendered by the UI tick. */
export function EventOverlays({ replay, uiT, overlays, names }: { replay: Replay; uiT: number; overlays: Overlays; names: Record<string, string> }) {
  const ann = replay.annotations;
  const passes = overlays.passes ? ann.passes.filter((p) => uiT >= p.t - 0.3 && uiT <= p.t + 2.6) : [];
  const strikes = overlays.angles ? ann.strikes.filter((s) => s.is_shot && Math.abs(uiT - s.t) < 1.6) : [];
  const scans = overlays.angles ? ann.scans.filter((s) => uiT >= s.t - 1.2 && uiT <= s.t + 0.6 && s.body_to_goal_deg !== undefined) : [];
  const playerPos = (pid: string, t: number): Vec3 => {
    const i = replay.tracking.player_ids.indexOf(pid);
    const f = Math.max(0, Math.min(replay.tracking.n_frames - 1, Math.round((t - replay.tracking.t_start) * replay.tracking.fps)));
    const p = replay.tracking.players[f][i];
    return [p[0], p[1], 0];
  };
  return (
    <group>
      {passes.map((p) => {
        const a = toWorld(p.start[0], p.start[1], 0.08), b = toWorld(p.end[0], p.end[1], 0.08);
        const col = diffColor(p.difficulty);
        const mid = a.clone().lerp(b, 0.5);
        const dir = b.clone().sub(a).normalize();
        const head = [b.clone().sub(dir.clone().multiplyScalar(1.2)).add(new THREE.Vector3(-dir.z, 0, dir.x).multiplyScalar(0.6)), b, b.clone().sub(dir.clone().multiplyScalar(1.2)).add(new THREE.Vector3(dir.z, 0, -dir.x).multiplyScalar(0.6))];
        return (
          <group key={p.event_id}>
            <Line points={[a, b]} color={col} lineWidth={3} dashed={p.outcome !== "complete"} dashSize={1} gapSize={0.6} />
            <Line points={head} color={col} lineWidth={3} />
            <Html position={[mid.x, 0.5, mid.z]} center zIndexRange={[6, 0]} style={{ pointerEvents: "none" }}>
              <div className="callout pass" style={{ borderColor: `#${col.getHexString()}` }}>
                {p.length_m.toFixed(0)} m · difficulty {(p.difficulty * 100).toFixed(0)}
                {p.speed_mps ? ` · ${(p.speed_mps * 3.6).toFixed(0)} km/h` : ""}
                {p.outcome !== "complete" ? " · lost" : ""}
              </div>
            </Html>
          </group>
        );
      })}
      {strikes.map((s) => {
        const [x, y] = playerPos(s.player_id, s.t);
        const shot = ann.shots.find((sh) => sh.event_id === s.event_id);
        return (
          <Html key={s.t} position={toWorld(x, y, 2.6).toArray()} center zIndexRange={[7, 0]} style={{ pointerEvents: "none" }}>
            <div className="callout strike">
              <div className="ttl">{names[s.player_id]} · strike ({s.kicking_foot})</div>
              <div>Knee {s.backswing_knee_deg.toFixed(0)}° → {s.contact_knee_deg.toFixed(0)}° at contact</div>
              <div>Hip {s.contact_hip_deg.toFixed(0)}° · trunk lean {s.trunk_lean_deg.toFixed(0)}° · support knee {s.support_knee_deg.toFixed(0)}°</div>
              {shot && <div>Shot {(shot.speed_mps * 3.6).toFixed(0)} km/h · xG {shot.xg.toFixed(2)}</div>}
            </div>
          </Html>
        );
      })}
      {scans.map((s) => {
        const [x, y] = playerPos(s.player_id, s.t);
        return (
          <Html key={`${s.player_id}-${s.t}`} position={toWorld(x, y, 2.4).toArray()} center zIndexRange={[7, 0]} style={{ pointerEvents: "none" }}>
            <div className="callout scan">
              {names[s.player_id]}: {s.scans} scan{s.scans === 1 ? "" : "s"} · body {s.body_to_goal_deg?.toFixed(0)}° to goal {s.open_body ? "(open)" : "(closed)"}
            </div>
          </Html>
        );
      })}
    </group>
  );
}

function CameraRig({ replay, clock, camera, focusId, script }: Ctx) {
  const { camera: cam } = useThree();
  const look = useRef(new THREE.Vector3());
  const shot = replay.annotations.shots[replay.annotations.shots.length - 1];
  useFrame((_, dt) => {
    const f = clock.frame;
    const t = replay.tracking.t_start + f / replay.tracking.fps;
    // playback clock lives here so it advances in lockstep with rendering
    if (clock.playing) {
      let rate = clock.rate;
      const slow = script?.slowmo.find((s) => t >= s.t_start && t <= s.t_end);
      if (clock.autoSlow && slow) rate = Math.min(rate, slow.rate);
      else if (clock.autoSlow && !script && shot && t > shot.t - 0.7 && t < shot.t + 0.6) rate = Math.min(rate, 0.3);
      clock.frame = Math.min(clock.frame + dt * replay.tracking.fps * rate, replay.tracking.n_frames - 1);
      if (clock.frame >= replay.tracking.n_frames - 1) clock.playing = false;
    }
    if (camera === "orbit") return;
    const ball = sampleBall(replay, f);
    let mode: CameraMode = camera;
    let target_id = focusId;
    if (camera === "director") {
      // follow the Replay Director agent's camera script when available
      const cue = script?.cameras.filter((c) => c.t <= t).slice(-1)[0];
      mode = cue ? (cue.mode as CameraMode) : shot && t > shot.t - 2.6 && t < shot.t + 1.2 ? "behind" : "broadcast";
      if (cue?.target) target_id = cue.target;
    }
    const desired = new THREE.Vector3();
    const target = ball.clone();
    if (mode === "broadcast") {
      desired.set(ball.x * 0.85, 21, 52);
      target.set(ball.x, 0, ball.z * 0.6);
    } else if (mode === "behind") {
      const goalX = shot ? (shot.end[0] > 52.5 ? 52.5 : -52.5) : 52.5;
      const dir = new THREE.Vector3(goalX - ball.x, 0, -ball.z).normalize();
      desired.copy(ball).sub(dir.clone().multiplyScalar(11)).setY(3.6);
      target.set(goalX, 1, 0).lerp(ball, 0.35);
    } else if (mode === "player") {
      const id = target_id ?? shot?.player_id;
      const i = id ? replay.tracking.player_ids.indexOf(id) : -1;
      if (i >= 0) {
        const fr = replay.tracking.players;
        const fi = Math.max(0, Math.min(fr.length - 1, Math.round(f)));
        const p = fr[fi][i], q = fr[Math.max(0, fi - 8)][i];
        const pw = toWorld(p[0], p[1]);
        let dir = new THREE.Vector3(p[0] - q[0], 0, -(p[1] - q[1]));
        if (dir.length() < 0.5) dir = ball.clone().sub(pw).setY(0);
        dir.normalize();
        desired.copy(pw).sub(dir.multiplyScalar(5.5)).setY(2.4);
        target.copy(pw).lerp(ball, 0.5).setY(1.2);
      }
    }
    const k = 1 - Math.exp(-dt * 3.2);
    cam.position.lerp(desired, k);
    look.current.lerp(target, k);
    cam.lookAt(look.current);
  });
  return camera === "orbit" ? <OrbitControls target={sampleBall(replay, clock.frame)} maxPolarAngle={Math.PI / 2.1} /> : null;
}

export default function ReplayScene(props: Ctx) {
  return (
    <>
      <hemisphereLight args={["#bcd6ff", "#0c1a10", 0.55]} />
      <directionalLight position={[-30, 60, 25]} intensity={2.2} castShadow shadow-mapSize={[2048, 2048]} shadow-camera-left={-70} shadow-camera-right={70} shadow-camera-top={50} shadow-camera-bottom={-50} />
      <directionalLight position={[40, 40, -30]} intensity={0.6} />
      <fog attach="fog" args={["#05070b", 90, 220]} />
      <Players {...props} />
      <Ball {...props} />
      <TacticalOverlays {...props} />
      <CameraRig {...props} />
    </>
  );
}
