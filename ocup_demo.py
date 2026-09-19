"""
Ocup-style Football Match Analysis — Demo
Motore statistico: Poisson / Dixon-Coles semplificato su dati sintetici.
Da migliorare in seguito collegando dati reali (football-data.co.uk, ecc.)
"""

import streamlit as st
import numpy as np
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# DATI DI ESEMPIO (sintetici, solo a scopo dimostrativo)
# attack: gol segnati attesi rispetto alla media di lega (1.0 = media)
# defense: gol subiti attesi rispetto alla media di lega (1.0 = media, <1 = difesa solida)
# ---------------------------------------------------------------------------
LEAGUE_AVG_GOALS = 1.35  # gol attesi "base" per squadra in una partita media

TEAMS = {
    "Real Madrid":       {"attack": 1.55, "defense": 0.75, "home_adv": 1.12},
    "Barcelona":         {"attack": 1.60, "defense": 0.80, "home_adv": 1.10},
    "Atletico Madrid":   {"attack": 1.25, "defense": 0.65, "home_adv": 1.08},
    "Manchester City":   {"attack": 1.70, "defense": 0.70, "home_adv": 1.10},
    "Liverpool":         {"attack": 1.58, "defense": 0.78, "home_adv": 1.12},
    "Arsenal":           {"attack": 1.50, "defense": 0.72, "home_adv": 1.10},
    "Manchester United": {"attack": 1.30, "defense": 0.95, "home_adv": 1.09},
    "Chelsea":           {"attack": 1.35, "defense": 0.88, "home_adv": 1.08},
    "Bayern Munich":     {"attack": 1.75, "defense": 0.68, "home_adv": 1.13},
    "Borussia Dortmund": {"attack": 1.45, "defense": 0.95, "home_adv": 1.11},
    "Bayer Leverkusen":  {"attack": 1.48, "defense": 0.82, "home_adv": 1.09},
    "Inter":             {"attack": 1.42, "defense": 0.72, "home_adv": 1.10},
    "AC Milan":          {"attack": 1.30, "defense": 0.80, "home_adv": 1.08},
    "Juventus":          {"attack": 1.25, "defense": 0.70, "home_adv": 1.09},
    "Napoli":            {"attack": 1.38, "defense": 0.78, "home_adv": 1.10},
    "Paris Saint-Germain": {"attack": 1.80, "defense": 0.72, "home_adv": 1.14},
    "Marseille":         {"attack": 1.20, "defense": 0.98, "home_adv": 1.09},
    "Ajax":              {"attack": 1.40, "defense": 0.90, "home_adv": 1.10},
    "Benfica":           {"attack": 1.45, "defense": 0.80, "home_adv": 1.11},
    "Porto":             {"attack": 1.35, "defense": 0.78, "home_adv": 1.10},
}

MAX_GOALS = 6  # limite matrice risultati esatti


# ---------------------------------------------------------------------------
# MOTORE STATISTICO
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
        matrix[i, j]
        for i in range(MAX_GOALS + 1)
        for j in range(MAX_GOALS + 1)
        if i + j > 2.5
    ) / matrix.sum()
    return btts, over_25


def confidence_volatility(home_p, draw_p, away_p):
    probs = np.array([home_p, draw_p, away_p])
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = -np.log(1 / 3)
    volatility = entropy / max_entropy  # 0 = esito quasi certo, 1 = massima incertezza
    confidence = 1 - volatility
    return confidence, volatility


def tactical_narrative(home, away, lh, la):
    h, a = TEAMS[home], TEAMS[away]
    lines = []
    if h["attack"] > a["defense"] + 0.5:
        lines.append(f"L'attacco di {home} ({h['attack']:.2f}) trova spazi contro una difesa di {away} non irresistibile ({a['defense']:.2f}).")
    if a["attack"] > h["defense"] + 0.5:
        lines.append(f"{away} può sfruttare in ripartenza una difesa di {home} non impenetrabile.")
    if h["defense"] < 0.75:
        lines.append(f"{home} mostra una fase difensiva solida in questa stagione simulata.")
    if a["defense"] < 0.75:
        lines.append(f"{away} concede pochissimo, gara che potrebbe restare bloccata.")
    if abs(lh - la) < 0.2:
        lines.append("Equilibrio marcato negli expected goals: partita dall'esito aperto.")
    if not lines:
        lines.append("Nessun fattore tattico dominante emerge dai dati sintetici disponibili.")
    return lines


# ---------------------------------------------------------------------------
# UI STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ocup Demo — Football Match Analysis", page_icon="⚽", layout="centered")

st.title("⚽ Football Match Analysis — Demo")
st.caption("Demo con dati sintetici · motore Poisson/Dixon-Coles semplificato · solo analisi, nessun consiglio di scommessa")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Squadra di casa", sorted(TEAMS.keys()), index=0)
with col2:
    away_options = [t for t in sorted(TEAMS.keys()) if t != home_team]
    away_team = st.selectbox("Squadra ospite", away_options, index=0)

if st.button("Analizza partita", type="primary"):
    lh, la = expected_goals(home_team, away_team)
    matrix = score_matrix(lh, la)
    home_p, draw_p, away_p = match_outcome_probs(matrix)
    top5 = top_scorelines(matrix)
    btts, over25 = btts_over_under(matrix)
    confidence, volatility = confidence_volatility(home_p, draw_p, away_p)
    tactics = tactical_narrative(home_team, away_team, lh, la)

    st.divider()
    st.subheader(f"{home_team} vs {away_team}")

    # Risultato più probabile
    best_score, best_prob = top5[0]
    st.markdown(f"**Risultato più probabile:** {best_score[0]}-{best_score[1]} ({best_prob*100:.1f}%)")

    # Probabilità 1X2
    st.markdown("**Probabilità esito (1X2)**")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} (1)", f"{home_p*100:.1f}%")
    c2.metric("Pareggio (X)", f"{draw_p*100:.1f}%")
    c3.metric(f"{away_team} (2)", f"{away_p*100:.1f}%")

    # 5 scenari di risultato
    st.markdown("**5 scenari di risultato più probabili**")
    for (i, j), p in top5:
        st.write(f"- {i}-{j}  →  {p*100:.1f}%")

    # xG
    st.markdown("**Gol attesi (xG)**")
    st.write(f"{home_team}: {lh}  ·  {away_team}: {la}")

    # BTTS / Over-Under
    st.markdown("**Altri mercati**")
    st.write(f"Entrambe le squadre segnano (BTTS): {btts*100:.1f}%")
    st.write(f"Over 2.5 gol: {over25*100:.1f}%")

    # Fattori tattici
    st.markdown("**Fattori tattici principali**")
    for line in tactics:
        st.write(f"- {line}")

    # Player updates (placeholder demo)
    st.markdown("**Aggiornamenti su giocatori e disponibilità**")
    st.info("Demo: nessuna fonte dati infortuni/formazioni collegata. In una versione reale qui comparirebbero assenze, squalifiche e formazioni previste.")

    # Confidenza / volatilità
    st.markdown("**Confidenza e volatilità**")
    st.write(f"Livello di confidenza del modello: {confidence*100:.0f}%")
    st.write(f"Volatilità dell'esito: {volatility*100:.0f}%")

    st.caption("⚠️ Solo analisi statistica. Nessun consiglio su scommesse o gioco d'azzardo.")
else:
    st.info("Seleziona due squadre e premi **Analizza partita**.")
