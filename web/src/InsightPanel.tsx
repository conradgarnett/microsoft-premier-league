import { useState } from "react";
import type { Claim, Insights, KeyMoment, MatchEvent, Replay, Team } from "./api";
import RhythmChart from "./RhythmChart";

type Tab = "moment" | "recap" | "agents";
const UI: Record<string, Record<string, string>> = {
  en: { moment: "Moment", recap: "Recap", agents: "Agents", why: "Why it matters", passes: "Passes in this replay", stats: "Match stats", loading: "Agents are analysing…" },
  es: { moment: "Jugada", recap: "Crónica", agents: "Agentes", why: "Por qué importa", passes: "Pases en esta repetición", stats: "Estadísticas", loading: "Los agentes están analizando…" },
  fr: { moment: "Action", recap: "Résumé", agents: "Agents", why: "Pourquoi c'est important", passes: "Passes de ce ralenti", stats: "Statistiques", loading: "Les agents analysent…" },
  pt: { moment: "Lance", recap: "Resumo", agents: "Agentes", why: "Por que importa", passes: "Passes neste replay", stats: "Estatísticas", loading: "Os agentes estão analisando…" },
};

function ClaimRow({ c, analyst, onSeek }: { c: Claim; analyst: boolean; onSeek: (id: string) => void }) {
  return (
    <li className="claim">
      <span>{c.text}</span>
      {analyst && (
        <span className="chips">
          {c.evidence.slice(0, 4).map((id) => (
            <button key={id} className="chip-ev" onClick={() => onSeek(id)} title="Jump to this event">{id}</button>
          ))}
          {c.evidence.length > 4 && <span className="more">+{c.evidence.length - 4}</span>}
        </span>
      )}
    </li>
  );
}

const PIPELINE = ["Replay Director", "Tactical Analyst", "Commentator", "Recap Writer", "Translator"];

export default function InsightPanel(props: {
  insights: Insights | null; moment: KeyMoment | null; mode: "fan" | "analyst"; lang: string; replay: Replay;
  names: Record<string, string>; metrics: Record<string, any> | null; homeId: string; awayId: string; events: MatchEvent[]; teams: Team[];
  onSeek: (t: number) => void;
}) {
  const { insights, moment, mode, lang, replay, names, metrics, homeId, awayId } = props;
  const [tab, setTab] = useState<Tab>("moment");
  const ui = UI[lang] ?? UI.en;
  const analyst = mode === "analyst";
  const evT = (id: string) => props.events.find((e) => e.id === id)?.t;
  const seek = (id: string) => { const t = evT(id); if (t !== undefined) props.onSeek(t); };
  const m = moment && insights?.moments[moment.id];

  return (
    <aside className="panel">
      <div className="tabs" role="tablist">
        {(["moment", "recap", "agents"] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{ui[t]}</button>
        ))}
      </div>

      {tab === "moment" && moment && (
        <>
          <div className="stats">
            <div><b>{moment.xg.toFixed(2)}</b><span>xG</span></div>
            <div><b>{moment.build_up_event_ids.length}</b><span>build-up</span></div>
            <div><b>#{moment.rank}</b><span>importance</span></div>
          </div>
          {!m && <p className="muted">{ui.loading}</p>}
          {m && (
            <>
              <h3 className="headline">{m.analysis.headline.text}</h3>
              {m.analysis.why_it_matters && (
                <div className="why"><span>{ui.why}</span>{m.analysis.why_it_matters.text}</div>
              )}
              <ul className="claims">
                {m.analysis.claims.map((c, i) => <ClaimRow key={i} c={c} analyst={analyst} onSeek={seek} />)}
              </ul>
            </>
          )}
          {analyst && (
            <div className="passes">
              <h4>{ui.passes}</h4>
              {replay.annotations.passes.map((p) => (
                <div key={p.event_id} className="pass-row" onClick={() => props.onSeek(p.t)}>
                  <span className="bar" style={{ width: `${p.difficulty * 100}%` }} />
                  <span>{names[p.player_id]} → {p.receiver_id ? names[p.receiver_id] : "✕"}</span>
                  <span>{p.length_m.toFixed(0)} m · {(p.difficulty * 100).toFixed(0)}</span>
                  <span className="evid">{p.event_id}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {tab === "recap" && (
        <>
          {!insights && <p className="muted">{ui.loading}</p>}
          {insights && (
            <>
              <h3 className="headline">{insights.recap.headline.text}</h3>
              <ul className="claims">
                {insights.recap.claims.map((c, i) => <ClaimRow key={i} c={c} analyst={analyst} onSeek={seek} />)}
              </ul>
            </>
          )}
          {metrics && metrics.rhythm?.length > 0 && (
            <>
              <h4>Game rhythm</h4>
              <RhythmChart bins={metrics.rhythm} homeId={homeId} awayId={awayId}
                homeColor={props.teams.find((t) => t.id === homeId)?.color_primary ?? "#fff"}
                awayColor={props.teams.find((t) => t.id === awayId)?.color_primary ?? "#888"} />
            </>
          )}
          {metrics && (
            <div className="team-table">
              <h4>{ui.stats}</h4>
              {(analyst ? ["possession_pct", "xg", "shots", "shots_on_target", "pass_completion_pct", "progressive_passes", "ppda", "field_tilt_pct", "defensive_line_m"] : ["possession_pct", "shots", "xg"]).map((k) => (
                <div key={k} className="row">
                  <span>{metrics.teams[homeId][k]}</span>
                  <span className="k">{k.replace(/_/g, " ").replace(" pct", " %")}</span>
                  <span>{metrics.teams[awayId][k]}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {tab === "agents" && insights && (
        <div className="agents">
          <p className="muted">Provider: <b>{insights.provider}</b> · {insights.trace.length} agent calls · every claim validated against event ids</p>
          <div className="pipeline">
            {PIPELINE.map((a, i) => {
              const calls = insights.trace.filter((t) => t.agent === a);
              const rejected = calls.filter((t) => t.status !== "ok").length;
              return (
                <div key={a} className={`stage-node ${calls.length ? "" : "idle"}`}>
                  <b>{a}</b>
                  <span>{calls.length ? `${calls.filter((t) => t.status === "ok").length} ok${rejected ? ` · ${rejected} retried` : ""}` : "not needed"}</span>
                  {i < PIPELINE.length - 1 && <i className="arrow">↓</i>}
                </div>
              );
            })}
          </div>
          <h4>Handoff log</h4>
          <ol className="trace">
            {insights.trace.slice(0, 40).map((t, i) => (
              <li key={i} className={t.status}>
                <span>{t.sender} → <b>{t.agent}</b></span>
                <span className="evid">{t.task} · {t.provider} · try {t.attempt} · {t.status}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </aside>
  );
}
