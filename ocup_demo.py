"""
Ocup-style Football Match Analysis — Multi-Campionato Reale
Motore statistico: Poisson su dati reali (da football-data.co.uk)
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# CONFIGURAZIONE E DIZIONARIO CAMPIONATI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ocup — Football Match Analysis Multi-League", page_icon="⚽", layout="centered")

st.title("⚽ Football Match Analysis — Dati Reali (Multi-Lega)")
st.caption("Seleziona il campionato, analizza le statistiche reali e calcola gli scenari con il modello di Poisson.")

# Dizionario dei campionati supportati con i relativi codici URL di football-data.co.uk (stagione corrente)
LEAGUES = {
    "🇮🇹 Serie A (Italia)": "I1",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Inghilterra)": "E0",
    "🇪🇸 La Liga (Spagna)": "SP1",
    "🇩🇪 Bundesliga (Germania)": "D1",
    "🇫🇷 Ligue 1 (Francia)": "F1",
    "🇳🇱 Eredivisie (Paesi Bassi)": "N1",
    "🇵🇹 Primeira Liga (Portogallo)": "P1"
}

# ---------------------------------------------------------------------------
# SELEZIONE CAMPIONATO E CARICAMENTO DATI
# ---------------------------------------------------------------------------
selected_league_name = st.selectbox("Seleziona il Campionato", list(LEAGUES.keys()))
league_code = LEAGUES[selected_league_name]

@st.cache_data(ttl=3600)
def load_football_data(code):
    # URL dinamico basato sul codice campionato (nota: 'mmz2526' o l'anno corrente della stagione attiva)
    url = f"https://www.football-data.co.uk/mmz2526/{code}.csv"
    try:
        df = pd.read_csv(url)
        df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        return df
    except Exception as e:
        return None

df_matches = load_football_data(league_code)

if df_matches is None or df_matches.empty:
    st.warning(f"Impossibile caricare i dati per {selected_league_name}. Potrebbero non essere disponibili o l'URL della stagione è in aggiornamento.")
    st.stop()

# ---------------------------------------------------------------------------
# ELABORAZIONE STATISTICA DELLE SQUADRE
# ---------------------------------------------------------------------------
def compute_team_stats(df):
    home_games = df[['HomeTeam', 'FTHG', 'FTAG']].rename(columns={'HomeTeam': 'Team', 'FTHG': 'GoalsFor', 'FTAG': 'GoalsAgainst'})
    away_games = df[['AwayTeam', 'FTAG', 'FTHG']].rename(columns={'AwayTeam': 'Team', 'FTAG': 'GoalsFor', 'FTHG': 'GoalsAgainst'})
    
    all_games = pd.concat([home_games, away_games])
    league_avg_scored = all_games['GoalsFor'].mean()
    
    teams_dict = {}
    unique_teams = sorted(all_games['Team'].unique())
    
    for team in unique_teams:
        t_data = all_games[all_games['Team'] == team]
        matches_played = len(t_data)
        if matches_played == 0:
            continue
        
        goals_scored_avg = t_data['GoalsFor'].mean()
        goals_conceded_avg = t_data['GoalsAgainst'].mean()
        
        attack = goals_scored_avg / league_avg_scored if league_avg_scored > 0 else 1.0
        defense = goals_conceded_avg / league_avg_scored if league_avg_scored > 0 else 1.0
        
        teams_dict[team] = {
            "attack": round(attack, 3),
            "defense": round(defense, 3),
            "home_adv": 1.10
        }
        
    return teams_dict, league_avg_scored

TEAMS, LEAGUE_AVG_GOALS = compute_team_stats(df_matches)
MAX_GOALS = 6

# ---------------------------------------------------------------------------
# MOTORE STATISTICO POISSON
# ---------------------------------------------------------------------------
def expected_goals(home, away):
    h = TEAMS[home]
    a = TEAMS[away]
    lambda_home = LEAGUE_AVG_GOALS * h["attack"] * a["defense"] * h["home_adv"]
    lambda_away = LEAGUE_AVG_GOALS * a["attack"] * h["defense"] / h["home_adv"]
    return round(lambda_home, 2), round(lambda_away, 2)

def score_matrix(lh, la):
    home_probs = [poisson.pmf(i, lh) for i in range(MAX_GOALS + 1)]
    away_probs = [poisson.pmf(i, la) for i in range(MAX_GOALS + 1)]
    matrix = np.outer(home_probs, away_probs)
    return matrix

def match_outcome_probs(matrix):
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total

def top_scorelines(matrix, n=5):
    flat = [((i, j), matrix[i, j]) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    flat.sort(key=lambda x: x[1], reverse=True)
    total = matrix.sum()
    return [(score, prob / total) for score, prob in flat[:n]]

def btts_over_under(matrix):
    btts = 1 - (matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0])
    over_25 = sum(
        matrix[i, j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i + j > 2.5
    ) / matrix.sum()
    return btts, over_25

def confidence_volatility(home_p, draw_p, away_p):
    probs = np.array([home_p, draw_p, away_p])
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = -np.log(1 / 3)
    volatility = entropy / max_entropy
    confidence = 1 - volatility
    return confidence, volatility

# ---------------------------------------------------------------------------
# INTERFACCIA UTENTE (UI)
# ---------------------------------------------------------------------------
st.divider()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Squadra di casa", sorted(TEAMS.keys()), index=0)
with col2:
    away_options = [t for t in sorted(TEAMS.keys()) if t != home_team]
    away_team = st.selectbox("Squadra ospite", away_options, index=0 if len(away_options) > 0 else None)

if st.button("Analizza partita", type="primary"):
    if home_team == away_team:
        st.error("Seleziona due squadre differenti!")
    else:
        lh, la = expected_goals(home_team, away_team)
        matrix = score_matrix(lh, la)
        home_p, draw_p, away_p = match_outcome_probs(matrix)
        top5 = top_scorelines(matrix)
        btts, over25 = btts_over_under(matrix)
        confidence, volatility = confidence_volatility(home_p, draw_p, away_p)

        st.divider()
        st.subheader(f"{home_team} vs {away_team} ({selected_league_name})")

        best_score, best_prob = top5[0]
        st.markdown(f"**Risultato più probabile:** {best_score[0]}-{best_score[1]} ({best_prob*100:.1f}%)")

        st.markdown("**Probabilità esito (1X2)**")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} (1)", f"{home_p*100:.1f}%")
        c2.metric("Pareggio (X)", f"{draw_p*100:.1f}%")
        c3.metric(f"{away_team} (2)", f"{away_p*100:.1f}%")

        st.markdown("**5 scenari di risultato più probabili**")
        for (i, j), p in top5:
            st.write(f"- {i}-{j}  →  {p*100:.1f}%")

        st.markdown("**Gol attesi (xG stimati dai dati reali della lega)**")
        st.write(f"{home_team}: {lh}  ·  {away_team}: {la}")

        st.markdown("**Altri mercati**")
        st.write(f"Entrambe le squadre segnano (BTTS): {btts*100:.1f}%")
        st.write(f"Over 2.5 gol: {over25*100:.1f}%")

        st.markdown("**Confidenza e volatilità del modello**")
        st.write(f"Confidenza: {confidence*100:.0f}%  ·  Volatilità: {volatility*100:.0f}%")

        st.caption("⚠️ Analisi statistica basata sui dati reali del campionato selezionato. Nessun consiglio di scommessa.")
else:
    st.info("Scegli il campionato, seleziona le squadre e premi **Analizza partita**.")
