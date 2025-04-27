import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random

st.set_page_config(page_title="Crypto Facile", page_icon=":money_with_wings:", layout="wide")

# --- Fonctions principales ---

def profile_section():
    st.subheader("👤 Profil de l'utilisateur")
    name = st.text_input("Nom", "Utilisateur")
    email = st.text_input("Email", "exemple@email.com")
    age = st.number_input("Âge", min_value=0, value=25)
    st.write(f"Bienvenue, **{name}**!")

def portfolio_section(df):
    st.subheader("📁 Portefeuille")
    st.dataframe(df)

def budget_section():
    st.subheader("💸 Gestion du Budget")
    revenus = st.number_input("Revenus mensuels ($)", min_value=0.0, value=3000.0)
    depenses = st.number_input("Dépenses mensuelles ($)", min_value=0.0, value=2000.0)
    epargne = revenus - depenses
    st.write(f"💰 Épargne mensuelle estimée : **${epargne:.2f}**")

def analysis_section(df):
    st.subheader("📊 Analyse des Investissements")
    fig = px.pie(df, names='Actif', values='Valeur', title="Répartition du portefeuille")
    st.plotly_chart(fig)

def simulator():
    st.subheader("🧮 Simulateur de Croissance Crypto")
    montant_initial = st.number_input("Montant initial ($)", min_value=0.0, value=1000.0)
    taux_croissance = st.slider("Taux de croissance attendu (%)", min_value=-100, max_value=300, value=50)
    duree = st.slider("Durée de l'investissement (années)", min_value=1, max_value=10, value=5)

    montant_final = montant_initial * ((1 + taux_croissance / 100) ** duree)
    st.write(f"📈 Montant final après {duree} ans : **${montant_final:,.2f}**")

    data = {"Année": list(range(duree + 1)),
            "Montant": [montant_initial * ((1 + taux_croissance / 100) ** i) for i in range(duree + 1)]}
    fig = px.line(data, x="Année", y="Montant", markers=True)
    st.plotly_chart(fig)

def compound_interest_simulator():
    st.subheader("📈 Simulateur Avancé d'Intérêts Composés")
    
    principal = st.number_input("Montant initial ($)", min_value=0.0, value=1000.0)
    annual_rate = st.number_input("Taux d'intérêt annuel (%)", min_value=0.0, value=5.0)
    years = st.number_input("Durée (années)", min_value=1, value=10)
    contribution = st.number_input("Contribution régulière ($)", min_value=0.0, value=100.0)
    contribution_frequency = st.selectbox("Fréquence de contribution", ["Mensuelle", "Trimestrielle", "Annuelle"])
    compounding_frequency = st.selectbox("Fréquence de capitalisation", ["Mensuelle", "Trimestrielle", "Annuelle"])

    freq_dict = {"Annuelle": 1, "Trimestrielle": 4, "Mensuelle": 12}
    n_contrib = freq_dict[contribution_frequency]
    n_compound = freq_dict[compounding_frequency]

    amount = principal
    timeline = []
    amounts = []

    total_periods = years * n_compound
    rate_per_period = (annual_rate / 100) / n_compound

    for period in range(total_periods + 1):
        if period > 0:
            if (period * n_contrib) % n_compound == 0:
                amount += contribution
            amount *= (1 + rate_per_period)
        timeline.append(period / n_compound)
        amounts.append(amount)

    final_amount = amounts[-1]
    total_invested = principal + contribution * (years * n_contrib)
    total_interest = final_amount - total_invested

    st.write(f"💰 Montant final : **${final_amount:,.2f}**")
    st.write(f"💸 Capital investi : **${total_invested:,.2f}**")
    st.write(f"📈 Intérêts gagnés : **${total_interest:,.2f}**")

    fig = px.line(x=timeline, y=amounts, labels={'x': 'Années', 'y': 'Montant ($)'}, title="Croissance du Capital avec Contributions")
    st.plotly_chart(fig)

    data = {"Année": [], "Montant ($)": []}
    for i in range(0, len(timeline)):
        if i % n_compound == 0:
            data["Année"].append(int(timeline[i]))
            data["Montant ($)"].append(amounts[i])

    df_summary = pd.DataFrame(data)
    st.dataframe(df_summary.style.format({"Montant ($)": "{:,.2f}"}))

def trading_section(df):
    st.subheader("💹 Simulation de Trading")
    actifs = df['Actif'].tolist()
    choix = st.multiselect("Choisissez des actifs à trader :", actifs)
    if choix:
        st.write("Trading simulation coming soon... 🚀")

def advice_section(df):
    st.subheader("🧠 Conseil Personnalisé")
    budget = st.number_input("Budget d'investissement disponible ($)", min_value=0.0, value=5000.0)
    nb_actifs = st.slider("Nombre d'actifs différents", min_value=1, max_value=10, value=3)
    selection = random.sample(df['Actif'].tolist(), nb_actifs)
    st.write(f"Nous vous conseillons d'investir dans : {', '.join(selection)}")

# --- Main App ---

def main():
    st.title("🚀 Crypto Facile")

    # Données fictives pour test
    data = {
        "Actif": ["Bitcoin", "Ethereum", "Cardano", "Solana", "Polkadot"],
        "Valeur": [5000, 3000, 1500, 2000, 1000]
    }
    df = pd.DataFrame(data)

    with st.sidebar:
        page = st.radio("Menu", ["Profil", "Portefeuille", "Budget", "Analyse", "Simulateur Crypto", "Simulateur Intérêts", "Trading", "Conseil"])

    if page == "Profil":
        profile_section()
    elif page == "Portefeuille":
        portfolio_section(df)
    elif page == "Budget":
        budget_section()
    elif page == "Analyse":
        analysis_section(df)
    elif page == "Simulateur Crypto":
        simulator()
    elif page == "Simulateur Intérêts":
        compound_interest_simulator()
    elif page == "Trading":
        trading_section(df)
    elif page == "Conseil":
        advice_section(df)

if __name__ == "__main__":
    main()
