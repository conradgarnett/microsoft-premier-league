"""Deterministic offline generation used when no model credentials are configured.

Each agent task is rendered from templates over the structured context the orchestrator
passes in. Output has the same JSON contract as the LLM path, including evidence ids,
plus a ``_render`` list (template key + slots) per line so the offline Translator can
re-render the same facts in another language instead of pretending to translate prose.
"""

from __future__ import annotations

import json
from typing import Any

LANGS = ("en", "es", "fr", "pt")

T: dict[str, dict[str, str]] = {
    # play-by-play ---------------------------------------------------------------
    "c_pass": {"en": "{p} finds {r}.", "es": "{p} encuentra a {r}.", "fr": "{p} trouve {r}.", "pt": "{p} encontra {r}."},
    "c_long": {"en": "{p} goes long towards {r}.", "es": "{p} busca en largo a {r}.", "fr": "{p} cherche {r} en profondeur.", "pt": "{p} lança longo para {r}."},
    "c_through": {"en": "{p} slides a through ball into {r}'s path!", "es": "¡{p} filtra un pase al hueco para {r}!", "fr": "{p} glisse une passe en profondeur pour {r} !", "pt": "{p} enfia a bola para {r}!"},
    "c_lost": {"en": "{p}'s pass is cut out.", "es": "El pase de {p} es interceptado.", "fr": "La passe de {p} est interceptée.", "pt": "O passe de {p} é interceptado."},
    "c_carry": {"en": "{p} drives forward.", "es": "{p} avanza con el balón.", "fr": "{p} avance balle au pied.", "pt": "{p} avança com a bola."},
    "c_press": {"en": "{p} closes down {o}.", "es": "{p} presiona a {o}.", "fr": "{p} presse {o}.", "pt": "{p} pressiona {o}."},
    "c_tackle": {"en": "Big tackle from {p}!", "es": "¡Gran entrada de {p}!", "fr": "Gros tacle de {p} !", "pt": "Grande desarme de {p}!"},
    "c_intercept": {"en": "{p} reads it and intercepts.", "es": "{p} lo lee e intercepta.", "fr": "{p} lit le jeu et intercepte.", "pt": "{p} lê a jogada e intercepta."},
    "c_shot": {"en": "{p} shoots from {d} metres...", "es": "{p} dispara desde {d} metros...", "fr": "{p} frappe à {d} mètres...", "pt": "{p} chuta de {d} metros..."},
    "c_goal": {"en": "GOAL! {p} scores! {home} {hs}-{as_} {away}.", "es": "¡GOL! ¡Marca {p}! {home} {hs}-{as_} {away}.", "fr": "BUT ! {p} marque ! {home} {hs}-{as_} {away}.", "pt": "GOL! {p} marca! {home} {hs}-{as_} {away}."},
    "c_saved": {"en": "Saved by {gk}!", "es": "¡Atajada de {gk}!", "fr": "Arrêt de {gk} !", "pt": "Defesa de {gk}!"},
    "c_off": {"en": "Wide of the target.", "es": "Se va desviado.", "fr": "C'est à côté.", "pt": "Para fora."},
    "c_blocked": {"en": "Blocked!", "es": "¡Bloqueado!", "fr": "Contré !", "pt": "Bloqueado!"},
    # analyst ----------------------------------------------------------------------
    "a_head_goal": {"en": "{p} converts a {xg} xG chance", "es": "{p} convierte una ocasión de {xg} xG", "fr": "{p} convertit une occasion à {xg} xG", "pt": "{p} converte uma chance de {xg} xG"},
    "a_head_miss": {"en": "{p}'s {xg} xG chance ends {outcome}", "es": "La ocasión de {p} ({xg} xG) termina {outcome}", "fr": "L'occasion de {p} ({xg} xG) se termine {outcome}", "pt": "A chance de {p} ({xg} xG) termina {outcome}"},
    "a_chance": {"en": "Shot from {d} m at {v} km/h; geometric xG {xg}.", "es": "Disparo desde {d} m a {v} km/h; xG geométrico {xg}.", "fr": "Frappe à {d} m, {v} km/h ; xG géométrique {xg}.", "pt": "Chute de {d} m a {v} km/h; xG geométrico {xg}."},
    "a_build": {"en": "Build-up: {n} on-ball actions in the possession, {prog} progressive pass(es).", "es": "Construcción: {n} acciones con balón en la posesión, {prog} pase(s) progresivo(s).", "fr": "Construction : {n} actions avec ballon, {prog} passe(s) progressive(s).", "pt": "Construção: {n} ações com bola na posse, {prog} passe(s) progressivo(s)."},
    "a_keypass": {"en": "Key pass by {p}: {len} m, difficulty {diff}/100, lane clearance {lane} m.", "es": "Pase clave de {p}: {len} m, dificultad {diff}/100, línea de pase libre {lane} m.", "fr": "Passe clé de {p} : {len} m, difficulté {diff}/100, couloir libre {lane} m.", "pt": "Passe decisivo de {p}: {len} m, dificuldade {diff}/100, linha livre {lane} m."},
    "a_pressure_on": {"en": "Shooter under pressure (nearest defender {pd} m).", "es": "Rematador bajo presión (defensor a {pd} m).", "fr": "Tireur sous pression (défenseur à {pd} m).", "pt": "Finalizador sob pressão (defensor a {pd} m)."},
    "a_pressure_off": {"en": "Shooter unpressured (nearest defender {pd} m).", "es": "Rematador sin presión (defensor a {pd} m).", "fr": "Tireur sans pression (défenseur à {pd} m).", "pt": "Finalizador sem pressão (defensor a {pd} m)."},
    "a_strike": {"en": "Strike: knee {bk}° at backswing to {ck}° at contact, hip {hip}°, trunk lean {tl}°.", "es": "Golpeo: rodilla {bk}° en el armado a {ck}° en el contacto, cadera {hip}°, inclinación {tl}°.", "fr": "Frappe : genou {bk}° à l'armé puis {ck}° au contact, hanche {hip}°, buste {tl}°.", "pt": "Chute: joelho {bk}° na preparação para {ck}° no contato, quadril {hip}°, tronco {tl}°."},
    "a_scan": {"en": "{p} scanned {n} time(s) before receiving, body {deg}° to goal ({open}).", "es": "{p} escaneó {n} vez/veces antes de recibir, cuerpo a {deg}° de la portería ({open}).", "fr": "{p} a scanné {n} fois avant de recevoir, corps à {deg}° du but ({open}).", "pt": "{p} escaneou {n} vez(es) antes de receber, corpo a {deg}° do gol ({open})."},
    "a_rhythm": {"en": "Phase context: control-vs-chaos index {c}/100, {ctrl} controlling the ball.", "es": "Contexto: índice control-caos {c}/100, {ctrl} controla el balón.", "fr": "Contexte : indice contrôle-chaos {c}/100, {ctrl} contrôle le ballon.", "pt": "Contexto: índice controle-caos {c}/100, {ctrl} controla a bola."},
    "open": {"en": "open", "es": "abierto", "fr": "ouvert", "pt": "aberto"},
    "closed": {"en": "closed", "es": "cerrado", "fr": "fermé", "pt": "fechado"},
    "o_saved": {"en": "in a save", "es": "en atajada", "fr": "sur un arrêt", "pt": "em defesa"},
    "o_off_target": {"en": "off target", "es": "fuera", "fr": "hors cadre", "pt": "para fora"},
    "o_blocked": {"en": "blocked", "es": "bloqueada", "fr": "contrée", "pt": "bloqueada"},
    # fan ----------------------------------------------------------------------------
    "f_head_goal": {"en": "{p} scores for {team}!", "es": "¡{p} marca para {team}!", "fr": "{p} marque pour {team} !", "pt": "{p} marca para o {team}!"},
    "f_head_miss": {"en": "So close for {p}!", "es": "¡Cerca para {p}!", "fr": "Si près pour {p} !", "pt": "Quase para {p}!"},
    "f_odds": {"en": "From that spot, a shot goes in about {pct} times out of 100.", "es": "Desde ahí, un disparo entra unas {pct} veces de cada 100.", "fr": "De là, une frappe finit au fond environ {pct} fois sur 100.", "pt": "Dali, um chute entra cerca de {pct} vezes em 100."},
    "f_build": {"en": "It started with {n} touches of patient build-up.", "es": "Empezó con {n} toques de construcción paciente.", "fr": "Tout est parti de {n} touches de construction patiente.", "pt": "Começou com {n} toques de construção paciente."},
    "f_build_fast": {"en": "It came from a quick break - just {n} touches.", "es": "Llegó de una contra rápida: solo {n} toques.", "fr": "Sur une attaque éclair : seulement {n} touches.", "pt": "Veio de um contra-ataque rápido: só {n} toques."},
    "f_keypass": {"en": "The killer ball came from {p}.", "es": "El pase decisivo fue de {p}.", "fr": "La passe décisive est venue de {p}.", "pt": "O passe mortal veio de {p}."},
    "f_power": {"en": "The ball flew at {v} km/h.", "es": "El balón voló a {v} km/h.", "fr": "Le ballon est parti à {v} km/h.", "pt": "A bola voou a {v} km/h."},
    "f_why_goal_lead": {"en": "Why it matters: {team} take the lead.", "es": "Por qué importa: {team} se pone por delante.", "fr": "Pourquoi c'est important : {team} prend l'avantage.", "pt": "Por que importa: o {team} assume a liderança."},
    "f_why_goal_level": {"en": "Why it matters: {team} draw level.", "es": "Por qué importa: {team} empata.", "fr": "Pourquoi c'est important : {team} égalise.", "pt": "Por que importa: o {team} empata."},
    "f_why_goal_extend": {"en": "Why it matters: {team} extend the lead.", "es": "Por qué importa: {team} amplía la ventaja.", "fr": "Pourquoi c'est important : {team} creuse l'écart.", "pt": "Por que importa: o {team} amplia a vantagem."},
    "f_why_goal_back": {"en": "Why it matters: {team} pull one back.", "es": "Por qué importa: {team} recorta distancias.", "fr": "Pourquoi c'est important : {team} réduit l'écart.", "pt": "Por que importa: o {team} diminui."},
    "f_why_miss": {"en": "Why it matters: a big chance that could have changed the game.", "es": "Por qué importa: una gran ocasión que pudo cambiar el partido.", "fr": "Pourquoi c'est important : une grosse occasion qui pouvait tout changer.", "pt": "Por que importa: uma grande chance que podia mudar o jogo."},
    "f_focus": {"en": "Your player {p} was involved in this move.", "es": "Tu jugador {p} participó en esta jugada.", "fr": "Votre joueur {p} a participé à l'action.", "pt": "Seu jogador {p} participou da jogada."},
    # recap ----------------------------------------------------------------------------
    "r_clip": {"en": "Tracked from real footage: {n} passes and {pc} changes of possession in {s} seconds.", "es": "Seguimiento de vídeo real: {n} pases y {pc} cambios de posesión en {s} segundos.", "fr": "Suivi sur vidéo réelle : {n} passes et {pc} changements de possession en {s} secondes.", "pt": "Rastreado de vídeo real: {n} passes e {pc} trocas de posse em {s} segundos."},
    "q_syn": {"en": "Synthetic match from a seeded engine (seed {seed}): all {n} events, positions and skeletons are simulated; movement is calibrated against aggregate real youth-football statistics.", "es": "Partido sintético de un motor con semilla ({seed}): los {n} eventos, posiciones y esqueletos son simulados; el movimiento está calibrado con estadísticas agregadas de fútbol juvenil real.", "fr": "Match synthétique issu d'un moteur à graine ({seed}) : les {n} événements, positions et squelettes sont simulés ; le mouvement est calibré sur des statistiques agrégées de football de jeunes réel.", "pt": "Partida sintética de um motor com semente ({seed}): os {n} eventos, posições e esqueletos são simulados; o movimento é calibrado com estatísticas agregadas de futebol juvenil real."},
    "q_vid": {"en": "Tracked from real video: {p} players, visible {vis}% of the time; ball located in {ball}% of frames.", "es": "Seguimiento de vídeo real: {p} jugadores, visibles el {vis}% del tiempo; balón localizado en el {ball}% de los fotogramas.", "fr": "Suivi sur vidéo réelle : {p} joueurs, visibles {vis}% du temps ; ballon localisé sur {ball}% des images.", "pt": "Rastreado de vídeo real: {p} jogadores, visíveis {vis}% do tempo; bola localizada em {ball}% dos quadros."},
    "q_events": {"en": "{n} events were detected automatically from the tracks; treat single passes as estimates (on synthetic ground truth the detector finds about 8 in 10 passes).", "es": "{n} eventos se detectaron automáticamente a partir del seguimiento; los pases individuales son estimaciones (con datos sintéticos de referencia el detector encuentra unos 8 de cada 10 pases).", "fr": "{n} événements ont été détectés automatiquement ; chaque passe est une estimation (sur vérité terrain synthétique, le détecteur trouve environ 8 passes sur 10).", "pt": "{n} eventos foram detectados automaticamente; passes individuais são estimativas (com verdade sintética o detector encontra cerca de 8 em 10 passes)."},
    "q_mocap": {"en": "Body motion is motion capture placed on the real tracks, not the observed pose.", "es": "El movimiento corporal es captura de movimiento colocada sobre las trayectorias reales, no la postura observada.", "fr": "Les mouvements du corps sont de la capture de mouvement posée sur les trajectoires réelles, pas la posture observée.", "pt": "O movimento corporal é captura de movimento aplicada às trajetórias reais, não a pose observada."},
    "q_lift": {"en": "Body poses are observed by one camera and lifted to 3D; depth is estimated, so joint angles are approximate.", "es": "Las posturas se observan con una cámara y se elevan a 3D; la profundidad es estimada, así que los ángulos son aproximados.", "fr": "Les postures sont vues par une seule caméra puis passées en 3D ; la profondeur est estimée, les angles sont donc approximatifs.", "pt": "As poses são observadas por uma câmera e elevadas a 3D; a profundidade é estimada, então os ângulos são aproximados."},
    "r_head": {"en": "{home} {hs}-{as_} {away}", "es": "{home} {hs}-{as_} {away}", "fr": "{home} {hs}-{as_} {away}", "pt": "{home} {hs}-{as_} {away}"},
    "r_result_win": {"en": "{w} beat {l} {ws}-{ls}.", "es": "{w} venció a {l} por {ws}-{ls}.", "fr": "{w} bat {l} {ws}-{ls}.", "pt": "{w} venceu o {l} por {ws}-{ls}."},
    "r_result_draw": {"en": "{home} and {away} drew {hs}-{as_}.", "es": "{home} y {away} empataron {hs}-{as_}.", "fr": "{home} et {away} se neutralisent {hs}-{as_}.", "pt": "{home} e {away} empataram em {hs}-{as_}."},
    "r_goal": {"en": "{minute}' {p} ({team}), {xg} xG.", "es": "{minute}' {p} ({team}), {xg} xG.", "fr": "{minute}' {p} ({team}), {xg} xG.", "pt": "{minute}' {p} ({team}), {xg} xG."},
    "r_xg": {"en": "On expected goals it was {home} {hx} - {ax} {away}; {better} created the better chances.", "es": "En goles esperados fue {home} {hx} - {ax} {away}; {better} generó las mejores ocasiones.", "fr": "Aux expected goals : {home} {hx} - {ax} {away} ; {better} s'est créé les meilleures occasions.", "pt": "Em gols esperados: {home} {hx} - {ax} {away}; o {better} criou as melhores chances."},
    "r_control": {"en": "{team} held {pct}% possession with a field tilt of {tilt}%.", "es": "{team} tuvo el {pct}% de posesión con un dominio territorial del {tilt}%.", "fr": "{team} a eu {pct}% de possession pour une domination territoriale de {tilt}%.", "pt": "O {team} teve {pct}% de posse e domínio territorial de {tilt}%."},
    "r_press": {"en": "{team} pressed hardest (PPDA {ppda}).", "es": "{team} presionó más (PPDA {ppda}).", "fr": "{team} a pressé le plus haut (PPDA {ppda}).", "pt": "O {team} pressionou mais (PPDA {ppda})."},
    "r_chaos": {"en": "The most chaotic spell came around minute {m} (chaos index {c}).", "es": "El tramo más caótico llegó hacia el minuto {m} (índice {c}).", "fr": "La période la plus chaotique est survenue vers la {m}e minute (indice {c}).", "pt": "O período mais caótico veio por volta do minuto {m} (índice {c})."},
    "r_focus": {"en": "{p}: {passes} passes ({comp}% completed), {shots} shot(s), {km} km covered, top speed {top} km/h.", "es": "{p}: {passes} pases ({comp}% completados), {shots} disparo(s), {km} km recorridos, velocidad máxima {top} km/h.", "fr": "{p} : {passes} passes ({comp}% réussies), {shots} tir(s), {km} km parcourus, pointe à {top} km/h.", "pt": "{p}: {passes} passes ({comp}% certos), {shots} chute(s), {km} km percorridos, velocidade máxima {top} km/h."},
}


def say(key: str, lang: str, **slots: Any) -> str:
    tpl = T[key].get(lang) or T[key]["en"]
    return tpl.format(**{k: (say(v[1:], lang) if isinstance(v, str) and v.startswith("@") else v) for k, v in slots.items()})


def line(key: str, evidence: list[str], lang: str, **slots: Any) -> dict[str, Any]:
    return {"text": say(key, lang, **slots), "evidence": evidence, "_render": {"key": key, "slots": slots}}


# --------------------------------------------------------------------------- tasks
def commentary(ctx: dict, lang: str) -> dict:
    n = ctx["names"]
    km = ctx["moment"]
    lines = []
    for e in ctx["build_up"][-4:]:
        p = n.get(e.get("player_id"), "?")
        if e["type"] == "pass":
            r = n.get(e.get("receiver_id") or e["attributes"].get("intended_receiver_id"), "?")
            key = "c_lost" if e.get("outcome") != "complete" else "c_through" if e["attributes"].get("through_ball") else "c_long" if e["attributes"].get("length_m", 0) > 30 else "c_pass"
            lines.append({"t": e["t"], **line(key, [e["id"]], lang, p=p, r=r)})
        elif e["type"] == "carry":
            lines.append({"t": e["t"], **line("c_carry", [e["id"]], lang, p=p)})
        elif e["type"] == "tackle":
            lines.append({"t": e["t"], **line("c_tackle", [e["id"]], lang, p=p)})
        elif e["type"] == "interception":
            lines.append({"t": e["t"], **line("c_intercept", [e["id"]], lang, p=p)})
    pe = ctx["pressure_events"][-1:] if ctx["pressure_events"] else []
    for e in pe:
        lines.append({"t": e["t"], **line("c_press", [e["id"]], lang, p=n.get(e["player_id"], "?"), o=n.get(e.get("opponent_id"), "?"))})
    s = ctx["shot_event"]
    lines.append({"t": s["t"], **line("c_shot", [s["id"]], lang, p=n[s["player_id"]], d=round(s["attributes"]["distance_m"]))})
    if km["kind"] == "goal":
        home_id = ctx["home_team_id"]
        away_id = next(t for t in ctx["teams"] if t != home_id)
        sc = km["score_after"]
        lines.append({"t": s["t"] + 1.0, **line("c_goal", km["evidence"][-1:], lang, p=n[s["player_id"]], home=ctx["teams"][home_id], away=ctx["teams"][away_id], hs=sc[home_id], as_=sc[away_id])})
    elif s.get("outcome") == "saved":
        lines.append({"t": s["t"] + 0.8, **line("c_saved", [s["id"]], lang, gk=n.get(s.get("opponent_id"), "the keeper"))})
    elif s.get("outcome") == "blocked":
        lines.append({"t": s["t"] + 0.4, **line("c_blocked", [s["id"]], lang)})
    else:
        lines.append({"t": s["t"] + 0.8, **line("c_off", [s["id"]], lang)})
    lines.sort(key=lambda x: x["t"])
    return {"lines": lines}


def _key_pass(ctx: dict) -> dict | None:
    pm = [p for p in ctx["pass_metrics"] if p["complete"]]
    return max(pm, key=lambda p: p["difficulty"]) if pm else None


def analysis(ctx: dict, lang: str, mode: str, focus: str | None) -> dict:
    n = ctx["names"]
    km = ctx["moment"]
    s = ctx["shot_event"]
    shot = ctx.get("shot") or {}
    ev_shot = [s["id"]]
    xg = f"{km['xg']:.2f}"
    kp = _key_pass(ctx)
    claims = []
    if mode == "analyst":
        head = line("a_head_goal", km["evidence"][-1:], lang, p=n[s["player_id"]], xg=xg) if km["kind"] == "goal" else line("a_head_miss", ev_shot, lang, p=n[s["player_id"]], xg=xg, outcome=f"@o_{s.get('outcome', 'saved')}")
        v = round((shot.get("speed_mps") or s["attributes"].get("ball_speed_mps", 0)) * 3.6)
        claims.append(line("a_chance", ev_shot, lang, d=s["attributes"]["distance_m"], v=v, xg=xg))
        prog = sum(1 for p in ctx["pass_metrics"] if p["progressive"])
        claims.append(line("a_build", km["build_up_event_ids"] or ev_shot, lang, n=len(km["build_up_event_ids"]), prog=prog))
        if kp:
            pe = next(e for e in ctx["build_up"] if e["id"] == kp["event_id"])
            claims.append(line("a_keypass", [kp["event_id"]], lang, p=n[pe["player_id"]], len=kp["length_m"], diff=round(kp["difficulty"] * 100), lane=kp["lane_clearance_m"]))
        pd = s["attributes"].get("shooter_pressure_m", 5)
        claims.append(line("a_pressure_on" if s.get("under_pressure") else "a_pressure_off", ev_shot, lang, pd=pd))
        st = ctx.get("strike")
        if st:
            claims.append(line("a_strike", ev_shot, lang, bk=round(st["backswing_knee_deg"]), ck=round(st["contact_knee_deg"]), hip=round(st["contact_hip_deg"]), tl=round(st["trunk_lean_deg"])))
        for sc in ctx["scans"][:1]:
            claims.append(line("a_scan", [sc["pass_event_id"]], lang, p=n[sc["player_id"]], n=sc["scans"], deg=round(sc["body_to_goal_deg"]), open="@open" if sc["open_body"] else "@closed"))
        rh = ctx["rhythm"]
        claims.append(line("a_rhythm", rh["evidence"][:3] or ev_shot, lang, c=rh["chaos_index"], ctrl=ctx["teams"][rh["controlling_team"]]))
        why = None
    else:
        team = ctx["teams"][s["team_id"]]
        head = line("f_head_goal", km["evidence"][-1:], lang, p=n[s["player_id"]], team=team) if km["kind"] == "goal" else line("f_head_miss", ev_shot, lang, p=n[s["player_id"]])
        claims.append(line("f_odds", ev_shot, lang, pct=max(1, round(km["xg"] * 100))))
        nb = len(km["build_up_event_ids"])
        claims.append(line("f_build" if nb >= 4 else "f_build_fast", km["build_up_event_ids"] or ev_shot, lang, n=max(nb, 1)))
        if kp:
            pe = next(e for e in ctx["build_up"] if e["id"] == kp["event_id"])
            claims.append(line("f_keypass", [kp["event_id"]], lang, p=n[pe["player_id"]]))
        v = shot.get("speed_mps") or s["attributes"].get("ball_speed_mps")
        if v:
            claims.append(line("f_power", ev_shot, lang, v=round(v * 3.6)))
        if km["kind"] == "goal":
            sc = km["score_after"]
            us, them = sc[s["team_id"]], next(v for k, v in sc.items() if k != s["team_id"])
            key = "f_why_goal_lead" if us == them + 1 else "f_why_goal_level" if us == them else "f_why_goal_extend" if us > them else "f_why_goal_back"
            why = line(key, km["evidence"][-1:], lang, team=team)
        else:
            why = line("f_why_miss", ev_shot, lang)
    if focus and focus in {e.get("player_id") for e in ctx["build_up"]} | {s["player_id"]}:
        claims.append(line("f_focus", [e["id"] for e in ctx["build_up"] if e.get("player_id") == focus][:3] or ev_shot, lang, p=n[focus]))
    return {"headline": head, "claims": claims, "why_it_matters": why}


def recap(ctx: dict, lang: str, mode: str, focus: str | None) -> dict:
    sm = ctx["summary"]
    teams = {t["id"]: t["name"] for t in sm["teams"]}
    home = sm["home_team_id"]
    away = next(t for t in teams if t != home)
    hs, as_ = sm["score"]["home"], sm["score"]["away"]
    ts = sm["team_stats"]
    # goalless matches still need evidence: fall back to the biggest chances, then rhythm events
    goal_ids = [g["event_id"] for g in sm["goals"]] or [k["shot_event_id"] for k in ctx["top_moments"]][:3] or ctx["rhythm_evidence"][:3] or ctx.get("event_evidence", [])[:3]
    claims = []
    if sm.get("source") == "video":  # a short real clip: describe what was tracked, not a "result"
        claims.append(line("r_clip", ctx.get("event_evidence", [])[:3], lang, n=ctx.get("clip_counts", {}).get("pass", 0), pc=ctx.get("clip_counts", {}).get("possession_change", 0), s=round(sm["duration_s"])))
    elif hs == as_:
        claims.append(line("r_result_draw", goal_ids, lang, home=teams[home], away=teams[away], hs=hs, as_=as_))
    else:
        w, l_ = (home, away) if hs > as_ else (away, home)
        claims.append(line("r_result_win", goal_ids, lang, w=teams[w], l=teams[l_], ws=max(hs, as_), ls=min(hs, as_)))
    for g in ctx["goal_moments"]:
        claims.append(line("r_goal", g["evidence"][-1:], lang, minute=g["minute"], p=ctx["names"][g["player_id"]], team=teams[g["team_id"]], xg=f"{g['xg']:.2f}"))
    better = home if ts[home]["xg"] >= ts[away]["xg"] else away
    shot_ids = [k["shot_event_id"] for k in ctx["top_moments"]][:4]
    claims.append(line("r_xg", shot_ids, lang, home=teams[home], away=teams[away], hx=ts[home]["xg"], ax=ts[away]["xg"], better=teams[better]))
    dom = home if ts[home]["possession_pct"] >= ts[away]["possession_pct"] else away
    claims.append(line("r_control", ctx["rhythm_evidence"][:3] or ctx.get("event_evidence", [])[:3], lang, team=teams[dom], pct=ts[dom]["possession_pct"], tilt=ts[dom]["field_tilt_pct"]))
    if mode == "analyst":
        presser = min(teams, key=lambda t: ts[t]["ppda"])
        claims.append(line("r_press", ctx["pressure_evidence"][:3], lang, team=teams[presser], ppda=ts[presser]["ppda"]))
        ch = max(ctx["rhythm"], key=lambda b: b["chaos_index"], default={"evidence": [], "t_start": 0, "chaos_index": 0})
        claims.append(line("r_chaos", ch["evidence"][:3], lang, m=int(ch["t_start"] // 60), c=ch["chaos_index"]))
    if focus and ctx.get("focus_stats"):
        f = ctx["focus_stats"]
        claims.append(line("r_focus", ctx.get("focus_evidence", [])[:3], lang, p=ctx["names"][focus], passes=f["passes"], comp=f["pass_completion_pct"], shots=f["shots"], km=f.get("distance_km", "?"), top=f.get("top_speed_kmh", "?")))
    claims = [c for c in claims if c["evidence"]]  # short video clips lack shots / pressure events: drop unsupported claims
    return {"headline": line("r_head", goal_ids, lang, home=teams[home], away=teams[away], hs=hs, as_=as_), "claims": claims}


def audit(ctx: dict, lang: str) -> dict:
    q = ctx["quality"]
    ev = q["evidence"]
    if q["source"] != "video":
        return {"claims": [line("q_syn", ev, lang, seed=q.get("seed"), n=q["n_events"])]}
    claims = [line("q_vid", ev, lang, p=q.get("tracked_players", 0), vis=q.get("player_visible_pct", 0), ball=q.get("ball_known_pct", 0))]
    if q["detected_events"]:
        claims.append(line("q_events", ev, lang, n=q["detected_events"]))
    claims.append(line("q_mocap" if q.get("skeleton_source") == "video-mocap" else "q_lift", ev, lang))
    return {"claims": claims}


def director(ctx: dict) -> dict:
    """Pick the reel (goals + top moments, chronological) and a camera script per moment."""
    moments = ctx["moments"]
    picks = [m for m in moments if m["kind"] == "goal"]
    picks += [m for m in sorted(moments, key=lambda m: m["rank"]) if m not in picks][: max(0, ctx.get("reel_size", 6) - len(picks))]
    reel = []
    for m in sorted(picks, key=lambda m: m["t"]):
        t = m["t"]
        cams = [
            {"t": m["replay"]["t_start"], "mode": "broadcast", "why": "establish shape and the build-up"},
            {"t": t - 2.6, "mode": "behind", "why": "show the shooting angle and the defenders in the lane"},
            {"t": t + 1.2, "mode": "player", "target": m["player_id"], "why": "isolate the shooter's technique"},
        ]
        slow = [{"t_start": t - 0.7, "t_end": t + 0.5, "rate": 0.3, "why": "ball strike"}]
        reel.append({"moment_id": m["id"], "cameras": cams, "slowmo": slow, "evidence": m["evidence"][-2:], "reason": f"{m['kind']} rank {m['rank']} (xG {m['xg']:.2f})"})
    return {"reel": reel}


def translate(payload: dict, lang: str) -> dict:
    """Re-render every templated line in ``lang``; free text without _render is left as-is."""

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            if "_render" in x:
                r = x["_render"]
                return {**x, "text": say(r["key"], lang, **r["slots"])}
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(payload)


def run_task(request: dict) -> str:
    """Entry point used by the offline chat client: request JSON in, response JSON out."""
    task, ctx = request["task"], request["context"]
    lang, mode, focus = request.get("lang", "en"), request.get("mode", "analyst"), request.get("focus")
    if task == "director":
        out = director(ctx)
    elif task == "commentary":
        out = commentary(ctx, lang)
    elif task == "analysis":
        out = analysis(ctx, lang, mode, focus)
    elif task == "recap":
        out = recap(ctx, lang, mode, focus)
    elif task == "audit":
        out = audit(ctx, lang)
    elif task == "translate":
        out = translate(ctx["payload"], lang)
    else:
        raise ValueError(f"unknown task {task}")
    return json.dumps(out, ensure_ascii=False)
