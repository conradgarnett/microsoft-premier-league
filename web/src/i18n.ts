// Viewer chrome in the four supported languages (agent text is localised server-side).
type Dict = Record<string, string>;

const EN: Dict = {
  goal: "Goal", big_chance: "Big chance", save: "Save", shot: "Shot", pass: "Pass", pressure: "Pressure",
  possession_change: "Turnover", carry: "Carry", tackle: "Tackle", interception: "Interception", foul: "Foul",
  free_kick: "Free kick", goal_kick: "Goal kick", clearance: "Clearance", ball_recovery: "Recovery", kickoff: "Kick-off",
  half_start: "Half start", half_end: "Half end",
  director: "Director", broadcast: "Broadcast", behind: "Behind ball", player: "Player cam", orbit: "Orbit",
  passes: "Pass lines", pressureOv: "Pressure", defLine: "Def. line", sprints: "Sprint trails", speeds: "Speeds", angles: "Body angles", tags: "Name tags",
  live: "Live", replay: "3D Replay", fan: "Fan", analyst: "Analyst", focus: "Focus player…", watch: "Watch the 3D replay ▸",
  feedSpeed: "Feed speed", restart: "Restart match", received: "events received", autoSlow: "auto slow-mo", loadingReplay: "Loading replay…",
};

const ES: Dict = {
  goal: "Gol", big_chance: "Ocasión clara", save: "Parada", shot: "Tiro", pass: "Pase", pressure: "Presión",
  possession_change: "Pérdida", carry: "Conducción", tackle: "Entrada", interception: "Intercepción", foul: "Falta",
  free_kick: "Tiro libre", goal_kick: "Saque de puerta", clearance: "Despeje", ball_recovery: "Recuperación", kickoff: "Saque inicial",
  half_start: "Inicio", half_end: "Descanso",
  director: "Realizador", broadcast: "Televisión", behind: "Tras el balón", player: "Cámara jugador", orbit: "Órbita",
  passes: "Líneas de pase", pressureOv: "Presión", defLine: "Línea defensiva", sprints: "Estelas de sprint", speeds: "Velocidades", angles: "Ángulos corporales", tags: "Nombres",
  live: "En vivo", replay: "Repetición 3D", fan: "Aficionado", analyst: "Analista", focus: "Jugador favorito…", watch: "Ver la repetición 3D ▸",
  feedSpeed: "Velocidad", restart: "Reiniciar partido", received: "eventos recibidos", autoSlow: "cámara lenta auto", loadingReplay: "Cargando repetición…",
};

const FR: Dict = {
  goal: "But", big_chance: "Grosse occasion", save: "Arrêt", shot: "Tir", pass: "Passe", pressure: "Pressing",
  possession_change: "Perte", carry: "Conduite", tackle: "Tacle", interception: "Interception", foul: "Faute",
  free_kick: "Coup franc", goal_kick: "Six mètres", clearance: "Dégagement", ball_recovery: "Récupération", kickoff: "Coup d'envoi",
  half_start: "Début", half_end: "Mi-temps",
  director: "Réalisateur", broadcast: "Télé", behind: "Derrière le ballon", player: "Caméra joueur", orbit: "Orbite",
  passes: "Lignes de passe", pressureOv: "Pressing", defLine: "Ligne défensive", sprints: "Traces de sprint", speeds: "Vitesses", angles: "Angles du corps", tags: "Noms",
  live: "Direct", replay: "Ralenti 3D", fan: "Supporter", analyst: "Analyste", focus: "Joueur favori…", watch: "Voir le ralenti 3D ▸",
  feedSpeed: "Vitesse", restart: "Rejouer le match", received: "événements reçus", autoSlow: "ralenti auto", loadingReplay: "Chargement du ralenti…",
};

const PT: Dict = {
  goal: "Gol", big_chance: "Grande chance", save: "Defesa", shot: "Chute", pass: "Passe", pressure: "Pressão",
  possession_change: "Perda", carry: "Condução", tackle: "Desarme", interception: "Interceptação", foul: "Falta",
  free_kick: "Tiro livre", goal_kick: "Tiro de meta", clearance: "Corte", ball_recovery: "Recuperação", kickoff: "Pontapé inicial",
  half_start: "Início", half_end: "Intervalo",
  director: "Diretor", broadcast: "TV", behind: "Atrás da bola", player: "Câmera jogador", orbit: "Órbita",
  passes: "Linhas de passe", pressureOv: "Pressão", defLine: "Linha defensiva", sprints: "Rastros de sprint", speeds: "Velocidades", angles: "Ângulos do corpo", tags: "Nomes",
  live: "Ao vivo", replay: "Replay 3D", fan: "Torcedor", analyst: "Analista", focus: "Jogador favorito…", watch: "Ver o replay 3D ▸",
  feedSpeed: "Velocidade", restart: "Reiniciar jogo", received: "eventos recebidos", autoSlow: "câmera lenta auto", loadingReplay: "Carregando replay…",
};

const ALL: Record<string, Dict> = { en: EN, es: ES, fr: FR, pt: PT };

export const tr = (lang: string, key: string): string => ALL[lang]?.[key] ?? EN[key] ?? key.replace("_", " ");
