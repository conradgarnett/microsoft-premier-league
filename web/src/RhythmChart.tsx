/** Game rhythm: xG per 5-minute bin (home up, away down) with the control-vs-chaos index as a line. */
interface Bin { t_start: number; t_end: number; chaos_index: number; home_possession: number; xg: Record<string, number> }

export default function RhythmChart({ bins, homeId, awayId, homeColor, awayColor }: { bins: Bin[]; homeId: string; awayId: string; homeColor: string; awayColor: string }) {
  const W = 300, H = 130, mid = 70, bw = W / bins.length;
  const maxXg = Math.max(0.3, ...bins.flatMap((b) => [b.xg[homeId], b.xg[awayId]]));
  const y = (v: number) => (v / maxXg) * 40;
  const line = bins.map((b, i) => `${i ? "L" : "M"}${(i + 0.5) * bw},${10 + (1 - b.chaos_index / 100) * 110}`).join(" ");
  return (
    <figure className="rhythm">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="xG per five minutes and control-versus-chaos index">
        {bins.map((b, i) => (
          <g key={i}>
            <rect x={i * bw} y={0} width={bw} height={H} fill={b.home_possession >= 0.5 ? homeColor : awayColor} opacity={0.06 + Math.abs(b.home_possession - 0.5) * 0.25} />
            <rect x={i * bw + 2} y={mid - y(b.xg[homeId])} width={bw - 4} height={y(b.xg[homeId])} fill={homeColor} rx={1.5} />
            <rect x={i * bw + 2} y={mid} width={bw - 4} height={y(b.xg[awayId])} fill={awayColor} rx={1.5} />
          </g>
        ))}
        <line x1={0} x2={W} y1={mid} y2={mid} stroke="rgba(255,255,255,0.25)" />
        <path d={line} fill="none" stroke="#ffe14d" strokeWidth={1.8} />
        <line x1={W / 2} x2={W / 2} y1={0} y2={H} stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
      </svg>
      <figcaption>
        <span><i style={{ background: homeColor }} />xG home</span>
        <span><i style={{ background: awayColor }} />xG away</span>
        <span><i style={{ background: "#ffe14d" }} />chaos index (up = chaotic)</span>
      </figcaption>
    </figure>
  );
}
