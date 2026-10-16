// Types mirror the pydantic schema in sim/schema.py and API responses in api/main.py.
export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export interface Player {
  id: string; name: string; short_name: string; number: number; position: string; team_id: string;
}
export interface Team {
  id: string; name: string; short_name: string; color_primary: string; color_secondary: string; formation: string; players: Player[];
}
export interface MatchEvent {
  id: string; t: number; period: number; minute: number; second: number; type: string;
  team_id: string | null; player_id: string | null; x: number | null; y: number | null;
  end_x: number | null; end_y: number | null; outcome: string | null; receiver_id: string | null;
  opponent_id: string | null; possession_id: number; under_pressure: boolean; related_ids: string[];
  attributes: Record<string, unknown>;
}
export interface MatchInfo {
  meta: { match_id: string; duration_s: number; half_length_s: number; home_team_id: string; away_team_id: string;
    final_score: { home: number; away: number }; source: string; home_attacks_right_first_half: boolean };
  teams: Team[];
  windows: { id: string; t_start: number; t_end: number; trigger_event_ids: string[] }[];
}
export interface KeyMoment {
  id: string; window_id: string; t: number; minute: number; kind: "goal" | "big_chance" | "save" | "shot";
  title: string; team_id: string; player_id: string; shot_event_id: string; outcome: string; xg: number;
  importance: number; rank: number; score_after: Record<string, number>; build_up_event_ids: string[];
  evidence: string[]; replay: { t_start: number; t_end: number };
}
export interface PassAnn {
  event_id: string; t: number; player_id: string; receiver_id: string | null; start: Vec2; end: Vec2;
  outcome: string; length_m: number; difficulty: number; progressive: boolean; speed_mps?: number;
}
export interface StrikeAnn {
  player_id: string; t: number; kicking_foot: string; backswing_knee_deg: number; contact_knee_deg: number;
  contact_hip_deg: number; follow_through_hip_deg: number; support_knee_deg: number; trunk_lean_deg: number;
  event_id?: string; is_shot: boolean;
}
export interface ScanAnn { player_id: string; t: number; scans: number; body_to_goal_deg?: number; open_body?: boolean }
export interface Replay {
  tracking: { id: string; t_start: number; fps: number; n_frames: number; player_ids: string[]; ball: Vec3[]; players: Vec2[][] };
  skeleton: { source?: string; t_start: number; fps: number; n_frames: number; joint_names: string[]; bones: [string, string][]; player_ids: string[]; frames: Vec3[][][] };
  annotations: {
    passes: PassAnn[];
    shots: { event_id: string; t: number; player_id: string; xg: number; outcome: string; start: Vec2; end: Vec3; speed_mps: number }[];
    strikes: StrikeAnn[]; scans: ScanAnn[];
    sprints: { player_id: string; t_start: number; t_end: number; top_speed_kmh: number }[];
    events: MatchEvent[];
  };
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

export const api = {
  matches: () => get<{ match_id: string; home: string; away: string }[]>("/api/matches"),
  match: (id: string) => get<MatchInfo>(`/api/matches/${id}`),
  events: (id: string) => get<MatchEvent[]>(`/api/matches/${id}/events`),
  moments: (id: string) => get<KeyMoment[]>(`/api/matches/${id}/key-moments`),
  metrics: (id: string) => get<Record<string, any>>(`/api/matches/${id}/metrics`),
  replay: (id: string, wid: string) => get<Replay>(`/api/matches/${id}/replay/${wid}`),
};

export interface Claim { text: string; evidence: string[] }
export interface CameraCue { t: number; mode: "broadcast" | "behind" | "player" | "orbit"; target?: string; why: string }
export interface ReelItem { moment_id: string; cameras: CameraCue[]; slowmo: { t_start: number; t_end: number; rate: number; why: string }[]; reason: string; evidence: string[] }
export interface Insights {
  provider: string; mode: string; lang: string; focus: string | null;
  director: { reel: ReelItem[] };
  moments: Record<string, { analysis: { headline: Claim; claims: Claim[]; why_it_matters: Claim | null }; commentary: { lines: (Claim & { t: number })[] } }>;
  recap: { headline: Claim; claims: Claim[] };
  trace: { agent: string; task: string; sender: string; provider: string; attempt: number; status: string; ms?: number; error?: string }[];
}

export const getInsights = (id: string, mode: string, lang: string, focus: string | null) =>
  get<Insights>(`/api/matches/${id}/insights?mode=${mode}&lang=${lang}${focus ? `&focus=${focus}` : ""}`);
