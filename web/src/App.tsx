import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, getInsights, type Insights, type KeyMoment, type MatchEvent, type MatchInfo, type Replay } from "./api";
import { tr } from "./i18n";
import InsightPanel from "./InsightPanel";
import Pitch from "./replay/Pitch";
import LiveScene, { type LiveFeed } from "./replay/LiveScene";
import ReplayScene, { EventOverlays, type CameraMode, type Clock, type Overlays } from "./replay/ReplayScene";

type Mode = "fan" | "analyst";
const fmtClock = (t: number) => `${Math.floor(t / 60)}'${String(Math.floor(t % 60)).padStart(2, "0")}`;

const OVERLAY_KEYS: [keyof Overlays, string][] = [
  ["passes", "passes"], ["pressure", "pressureOv"], ["defLine", "defLine"], ["sprints", "sprints"], ["speeds", "speeds"], ["angles", "angles"], ["tags", "tags"],
];
const DEFAULT_OVERLAYS: Record<Mode, Overlays> = {
  analyst: { passes: true, pressure: true, defLine: true, sprints: true, speeds: true, angles: true, tags: true },
  fan: { passes: true, pressure: false, defLine: false, sprints: true, speeds: false, angles: false, tags: true },
};

export default function App() {
  const [matchId, setMatchId] = useState<string | null>(null);
  const [info, setInfo] = useState<MatchInfo | null>(null);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [moments, setMoments] = useState<KeyMoment[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
  const [moment, setMoment] = useState<KeyMoment | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("analyst");
  const [overlays, setOverlays] = useState<Overlays>(DEFAULT_OVERLAYS.analyst);
  const [camera, setCamera] = useState<CameraMode>("director");
  const [focusId, setFocusId] = useState<string | null>(null);
  const [lang, setLang] = useState("en");
  const [insights, setInsights] = useState<Insights | null>(null);
  const [view, setView] = useState<"replay" | "live">("replay");
  const feed = useRef<LiveFeed>({ prev: null, next: null }).current;
  const [liveEvents, setLiveEvents] = useState<MatchEvent[]>([]);
  const [liveT, setLiveT] = useState(0);
  const liveTRef = useRef(0);
  const [liveSpeed, setLiveSpeed] = useState(20);
  const [toast, setToast] = useState<KeyMoment | null>(null);
  const [liveSession, setLiveSession] = useState(0);
  const clock = useRef<Clock>({ frame: 0, playing: true, rate: 1, autoSlow: true }).current;
  const [uiT, setUiT] = useState(0);
  const [, force] = useState(0);

  useEffect(() => {
    api.matches().then((ms) => setMatchId(ms[0]?.match_id ?? null)).catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    if (!matchId) return;
    Promise.all([api.match(matchId), api.events(matchId), api.moments(matchId), api.metrics(matchId)])
      .then(([i, ev, km, mt]) => {
        setInfo(i); setEvents(ev); setMoments(km); setMetrics(mt);
        const top = [...km].sort((a, b) => a.rank - b.rank)[0];
        // video-derived clips have no shots yet: replay the raw window instead
        const w = i.windows[0];
        setMoment(top ?? (w ? { id: `win-${w.id}`, window_id: w.id, t: w.t_start, minute: Math.floor(w.t_start / 60), kind: "shot", title: i.meta.source === "video" ? "Video clip (pose from footage)" : "Replay", team_id: i.meta.home_team_id, player_id: "", shot_event_id: "", outcome: "", xg: 0, importance: 0, rank: 1, score_after: {}, build_up_event_ids: [], evidence: [], replay: { t_start: w.t_start, t_end: w.t_end } } : null));
      })
      .catch((e) => setError(String(e)));
  }, [matchId]);
  useEffect(() => {
    if (!matchId || !moment) return;
    setLoading(true);
    api.replay(matchId, moment.window_id).then((r) => {
      setReplay(r); clock.frame = 0; clock.playing = true; setLoading(false);
    }).catch((e) => setError(String(e)));
  }, [matchId, moment, clock]);
  useEffect(() => {
    if (!matchId) return;
    setInsights(null);
    getInsights(matchId, mode, lang, focusId).then(setInsights).catch(() => setInsights(null));
  }, [matchId, mode, lang, focusId]);
  // live mode: simulated real-time feed over WebSocket (events + 2 fps positions)
  useEffect(() => {
    if (view !== "live" || !matchId) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/matches/${matchId}/live?speed=${liveSpeed}&start_t=${liveTRef.current}`);
    const pending: MatchEvent[] = [];
    ws.onmessage = (msg) => {
      const m = JSON.parse(msg.data);
      if (m.type === "frame") {
        feed.prev = feed.next;
        feed.next = { t: m.t, ball: m.ball, players: m.players, at: performance.now() };
        liveTRef.current = m.t;
      } else if (m.type === "event") {
        pending.push(m.event);
        if (m.event.type === "shot") {
          const km = moments.find((k) => k.shot_event_id === m.event.id);
          if (km && (km.kind === "goal" || km.kind === "big_chance" || km.rank <= 8)) setToast(km);
        }
      }
    };
    const id = setInterval(() => {
      setLiveT(liveTRef.current);
      if (pending.length) setLiveEvents((ev) => [...ev, ...pending.splice(0)]);
    }, 250);
    return () => { clearInterval(id); ws.close(); };
  }, [view, matchId, liveSpeed, moments, feed, liveSession]);
  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 12000);
    return () => clearTimeout(id);
  }, [toast]);
  // 10 Hz UI tick reads the scene clock
  useEffect(() => {
    const id = setInterval(() => replay && setUiT(replay.tracking.t_start + clock.frame / replay.tracking.fps), 100);
    return () => clearInterval(id);
  }, [replay, clock]);

  const names = useMemo(() => Object.fromEntries((info?.teams ?? []).flatMap((t) => t.players.map((p) => [p.id, p.short_name]))), [info]);
  const teamById = useMemo(() => Object.fromEntries((info?.teams ?? []).map((t) => [t.id, t])), [info]);
  const home = info?.teams.find((t) => t.id === info.meta.home_team_id);
  const away = info?.teams.find((t) => t.id === info.meta.away_team_id);
  const clockT = view === "live" ? liveT : uiT;
  const evSource = view === "live" ? liveEvents : events;
  const score = useMemo(() => {
    const s = { home: 0, away: 0 };
    for (const e of evSource) if (e.type === "goal" && e.t <= clockT) e.team_id === home?.id ? s.home++ : s.away++;
    return s;
  }, [evSource, clockT, home]);
  const ticker = useMemo(() => evSource.filter((e) => e.t <= clockT && e.t > clockT - (view === "live" ? 90 : 40) && !["possession_change", "carry"].includes(e.type)).slice(-7).reverse(), [evSource, clockT, view]);

  if (error) return <div className="splash">Could not reach the API: {error}</div>;
  if (!info || !replay || !home || !away) return <div className="splash">Loading match…</div>;

  const dur = info.meta.duration_s;
  const windowLen = replay.tracking.n_frames / replay.tracking.fps;
  const setMode2 = (m: Mode) => { setMode(m); setOverlays(DEFAULT_OVERLAYS[m]); };
  const reel = moment ? insights?.director.reel.find((r) => r.moment_id === moment.id) : undefined;
  const caption = moment ? insights?.moments[moment.id]?.commentary.lines.filter((l) => l.t <= uiT && l.t > uiT - 3.5).slice(-1)[0] : undefined;
  const seek = (t: number) => {
    clock.frame = Math.max(0, Math.min(replay.tracking.n_frames - 1, (t - replay.tracking.t_start - 1.5) * replay.tracking.fps));
    clock.playing = true;
    setUiT(t);
  };
  const shooter = moment ? teamById[moment.team_id] : null;

  return (
    <div className="app">
      <header className="top">
        <div className="scorebug">
          <span className="chip" style={{ background: home.color_primary }} />
          <span className="team">{home.short_name}</span>
          <span className="score">{score.home} – {score.away}</span>
          <span className="team">{away.short_name}</span>
          <span className="chip" style={{ background: away.color_primary }} />
          <span className="clock">{fmtClock(clockT)}</span>
          <span className={view === "live" ? "live-badge" : "replay-badge"}>{view === "live" ? "LIVE" : "REPLAY"}</span>
        </div>
        <div className="brand">Tactical Replay <small>{info.meta.source === "video" ? `real footage · tracked from video · motion ${replay.skeleton.source === "video-mocap" ? "mocap on real tracks" : "single-view pose lift"}` : "synthetic league"} · {info.meta.match_id}</small></div>
        <div className="controls-top">
          <div className="seg">
            <button className={view === "live" ? "on" : ""} onClick={() => setView("live")}>{tr(lang, "live")}</button>
            <button className={view === "replay" ? "on" : ""} onClick={() => setView("replay")}>{tr(lang, "replay")}</button>
          </div>
          <div className="seg">
            {(["fan", "analyst"] as Mode[]).map((m) => (
              <button key={m} className={mode === m ? "on" : ""} onClick={() => setMode2(m)}>{tr(lang, m)}</button>
            ))}
          </div>
          <select value={lang} onChange={(e) => setLang(e.target.value)} aria-label="Language">
            <option value="en">English</option>
            <option value="es">Español</option>
            <option value="fr">Français</option>
            <option value="pt">Português</option>
          </select>
          <select value={focusId ?? ""} onChange={(e) => setFocusId(e.target.value || null)} aria-label="Focus player">
            <option value="">{tr(lang, "focus")}</option>
            {info.teams.map((t) => (
              <optgroup key={t.id} label={t.name}>
                {t.players.map((p) => <option key={p.id} value={p.id}>{p.number} {p.name}</option>)}
              </optgroup>
            ))}
          </select>
        </div>
      </header>

      <main className="stage">
        <div className="viewport">
          <Canvas shadows camera={{ position: [0, 25, 55], fov: 32, near: 0.1, far: 500 }} gl={{ antialias: true, preserveDrawingBuffer: true }} dpr={[1, 2]}>
            <color attach="background" args={["#05070b"]} />
            <Pitch />
            {view === "live" ? (
              <LiveScene feed={feed} teams={info.teams} playerIds={replay.tracking.player_ids} />
            ) : (
              <>
                <ReplayScene replay={replay} teams={info.teams} clock={clock} overlays={overlays} camera={camera} focusId={focusId} script={reel} />
                <EventOverlays replay={replay} uiT={uiT} overlays={overlays} names={names} />
              </>
            )}
          </Canvas>
          {loading && view === "replay" && <div className="loading">{tr(lang, "loadingReplay")}</div>}
          {view === "live" && toast && (
            <button className="toast" onClick={() => { setMoment(toast); setView("replay"); setToast(null); }}>
              <span>{tr(lang, toast.kind)} · {toast.minute}'</span>
              <b>{toast.title}</b>
              <em>{tr(lang, "watch")}</em>
            </button>
          )}
          {view === "replay" && <div className="cams">
            {(["director", "broadcast", "behind", "player", "orbit"] as CameraMode[]).map((c) => (
              <button key={c} className={camera === c ? "on" : ""} onClick={() => setCamera(c)}>
                {tr(lang, c)}
              </button>
            ))}
          </div>}
          {view === "replay" && <div className="overlay-toggles">
            {OVERLAY_KEYS.map(([k, label]) => (
              <label key={k}><input type="checkbox" checked={overlays[k]} onChange={() => setOverlays({ ...overlays, [k]: !overlays[k] })} /> {tr(lang, label)}</label>
            ))}
          </div>}
          {view === "replay" && moment && (
            <div className="lower-third" style={{ borderColor: shooter?.color_primary }}>
              <div className="kind">{moment.id.startsWith("win-") ? tr(lang, "replay") : tr(lang, moment.kind)} · {moment.minute}'</div>
              <div className="title">{moment.player_id && names[moment.player_id] ? `${tr(lang, moment.kind)} – ${names[moment.player_id]} (${shooter?.short_name ?? ""})` : moment.title}</div>
            </div>
          )}
          {view === "replay" && caption && <div className="caption">{caption.text}</div>}
          <div className="ticker">
            {ticker.map((e) => (
              <div key={e.id} className={`tick ${e.type}`}>
                <span className="t">{fmtClock(e.t)}</span>
                <span className="dot" style={{ background: e.team_id ? teamById[e.team_id]?.color_primary : "#888" }} />
                {tr(lang, e.type)} {e.player_id ? names[e.player_id] : ""}
                {mode === "analyst" && <span className="evid">{e.id}</span>}
              </div>
            ))}
          </div>
        </div>

        <InsightPanel insights={insights} moment={moment} mode={mode} lang={lang} replay={replay} names={names} metrics={metrics}
          homeId={home.id} awayId={away.id} events={events} onSeek={seek} teams={info.teams} />
      </main>

      <footer className="bottom">
        {view === "live" ? (
          <div className="transport">
            <span className="tc">{tr(lang, "feedSpeed")}</span>
            <div className="seg">
              {[5, 20, 60].map((sp) => (
                <button key={sp} className={liveSpeed === sp ? "on" : ""} onClick={() => setLiveSpeed(sp)}>{sp}×</button>
              ))}
            </div>
            <button onClick={() => { liveTRef.current = 0; setLiveEvents([]); setLiveT(0); setLiveSession((n) => n + 1); }}>{tr(lang, "restart")}</button>
            <span className="tc">{evSource.length} {tr(lang, "received")}</span>
          </div>
        ) : (
        <div className="transport">
          <button onClick={() => { clock.playing = !clock.playing; force((n) => n + 1); }} aria-label="Play or pause">{clock.playing ? "❚❚" : "▶"}</button>
          <button onClick={() => { clock.frame = 0; clock.playing = true; force((n) => n + 1); }} aria-label="Restart">↺</button>
          <input type="range" min={0} max={replay.tracking.n_frames - 1} step={1} value={Math.round((uiT - replay.tracking.t_start) * replay.tracking.fps)}
            onChange={(e) => { clock.frame = Number(e.target.value); setUiT(replay.tracking.t_start + clock.frame / replay.tracking.fps); }} aria-label="Scrub" />
          <span className="tc">{(uiT - replay.tracking.t_start).toFixed(1)} / {windowLen.toFixed(1)} s</span>
          <div className="seg">
            {[0.1, 0.25, 0.5, 1, 2].map((r) => (
              <button key={r} className={clock.rate === r ? "on" : ""} onClick={() => { clock.rate = r; force((n) => n + 1); }}>{r}×</button>
            ))}
          </div>
          <label className="auto"><input type="checkbox" checked={clock.autoSlow} onChange={() => { clock.autoSlow = !clock.autoSlow; force((n) => n + 1); }} /> {tr(lang, "autoSlow")}</label>
        </div>
        )}
        <div className="timeline">
          <div className="half" style={{ left: `${(info.meta.half_length_s / dur) * 100}%` }} />
          {view === "live" && <div className="progress" style={{ width: `${(liveT / dur) * 100}%` }} />}
          {moments.filter((m) => view === "replay" || m.t <= liveT).map((m) => (
            <button key={m.id} className={`mk ${m.kind} ${moment?.id === m.id ? "sel" : ""}`} style={{ left: `${(m.t / dur) * 100}%`, borderColor: teamById[m.team_id]?.color_primary }}
              onClick={() => { setMoment(m); setView("replay"); }} title={`${m.minute}' ${m.title} (xG ${m.xg})`}>
              {m.kind === "goal" ? "G" : ""}
            </button>
          ))}
        </div>
      </footer>
    </div>
  );
}
