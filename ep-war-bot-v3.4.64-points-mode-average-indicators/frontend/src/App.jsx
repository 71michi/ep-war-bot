import { useEffect, useMemo, useState } from 'react'

const API_URL = 'http://localhost:8000/api/wars'

const MODE_META = {
  'Grad strzał': { icon: '➶', short: 'GS' },
  'Krwawa wojna': { icon: '🩸', short: 'KW' },
  'Horda nieumarłych': { icon: '☠', short: 'HN' },
  'Wojenne wyrównanie': { icon: '⚖', short: 'WW' },
  'Lepszy atak': { icon: '⬆', short: 'LA' },
  default: { icon: '⚔', short: 'W' },
}

function getModeMeta(mode) {
  return MODE_META[mode] || MODE_META.default
}

function scaledPoints(points, participants) {
  return Math.round((points * participants) / 30)
}

function AverageDivider({ average }) {
  return (
    <div className="average-divider" aria-label={`Średnia: ${average.toFixed(1)}`}>
      <div className="average-line" />
      <span>Średni wynik: {average.toFixed(1)}</span>
      <div className="average-line" />
    </div>
  )
}

function PlayerRow({ player, average, participants, compact = true }) {
  const modeMeta = getModeMeta(player.mode)
  const above = player.points >= average
  const diff = Math.abs(player.points - average).toFixed(1)
  const scaled = scaledPoints(player.points, participants)

  return (
    <div className={`player-row ${above ? 'above' : 'below'} ${compact ? 'compact' : ''}`}>
      <div className="player-main">
        <div className="player-name-wrap">
          <span className="player-name">{player.player}</span>
          <span className="mode-pill" title={player.mode}>
            <span className="mode-pill-icon">{modeMeta.icon}</span>
            <span className="mode-pill-text">{modeMeta.short}</span>
          </span>
        </div>
        <div className="player-score-wrap">
          <span className="player-score">{player.points}</span>
          <span className={`avg-indicator ${above ? 'up' : 'down'}`} title={above ? 'Powyżej średniej' : 'Poniżej średniej'}>
            {above ? '▲' : '▼'} {diff}
          </span>
        </div>
      </div>

      <div className="player-sub">
        <span>Przeliczone: {scaled}</span>
        <span>Ataki: {player.attacks_used ?? 6}/6</span>
        {player.roster_status === 'outside' ? <span className="outside-tag">Poza rosterem</span> : null}
      </div>
    </div>
  )
}

function WarCard({ war, opened, onToggle }) {
  const average = useMemo(() => {
    if (!war.players?.length) return 0
    return war.players.reduce((sum, player) => sum + player.points, 0) / war.players.length
  }, [war])

  const sortedPlayers = useMemo(() => {
    return [...war.players].sort((a, b) => b.points - a.points)
  }, [war])

  const splitIndex = sortedPlayers.findIndex((p) => p.points < average)

  const diff = war.alliance_points - war.opponent_points
  const warModeMeta = getModeMeta(war.mode)

  return (
    <section className="war-card">
      <button className="war-header" onClick={onToggle}>
        <div className="war-header-left">
          <div className="war-date">{war.date}</div>
          <div className="war-opponent">{war.opponent}</div>
        </div>

        <div className="war-header-mid">
          <span className={`result-chip ${war.result === 'Zwycięstwo' ? 'win' : war.result === 'Porażka' ? 'loss' : 'draw'}`}>
            {war.result}
          </span>
          <span className="mode-chip" title={war.mode}>{warModeMeta.icon} {war.mode}</span>
        </div>

        <div className="war-header-right">
          <div>{war.alliance_points} : {war.opponent_points}</div>
          <div className={`war-diff ${diff >= 0 ? 'positive' : 'negative'}`}>
            {diff >= 0 ? '+' : ''}{diff}
          </div>
        </div>
      </button>

      {opened ? (
        <div className="war-body">
          <div className="war-body-top">
            <span>Uczestnicy: {war.participant_count}/30</span>
            <span>Średnia punktów: {average.toFixed(1)}</span>
          </div>

          <AverageDivider average={average} />

          <div className="players-list">
            {sortedPlayers.map((player, index) => {
              const showMidDivider = splitIndex > 0 && index === splitIndex
              return (
                <div key={`${war.id}-${player.player}`}>
                  {showMidDivider ? (
                    <div className="mid-separator">
                      <span>Granica średniej</span>
                    </div>
                  ) : null}
                  <PlayerRow player={player} average={average} participants={war.participant_count} />
                </div>
              )
            })}
          </div>
        </div>
      ) : null}
    </section>
  )
}

export default function App() {
  const [wars, setWars] = useState([])
  const [openedWarId, setOpenedWarId] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    fetch(API_URL)
      .then((res) => {
        if (!res.ok) throw new Error('Nie udało się pobrać danych z API')
        return res.json()
      })
      .then((data) => {
        if (cancelled) return
        setWars(data)
        if (data[0]?.id) setOpenedWarId(data[0].id)
      })
      .catch((err) => {
        if (cancelled) return
        setError(err.message || 'Nieznany błąd')
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>EP War Bot</h1>
          <p>v3.4.64 • kompaktowe szczegóły wojny • tryb przy wyniku • średnia</p>
        </div>
      </header>

      {error ? <div className="error-box">{error}</div> : null}

      <main className="wars-grid">
        {wars.map((war) => (
          <WarCard
            key={war.id}
            war={war}
            opened={openedWarId === war.id}
            onToggle={() => setOpenedWarId((current) => current === war.id ? null : war.id)}
          />
        ))}
      </main>
    </div>
  )
}
