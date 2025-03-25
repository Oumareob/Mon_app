import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Configuration
st.set_page_config(page_title="Crypto Facile", layout="wide")

# Style CSS
st.markdown("""
    <style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    .trade-box {background-color: #2A2D3E; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .tip-box {background-color: #1E2130; padding: 10px; border-radius: 5px;}
    .analysis-box {background-color: #252736; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .budget-box {background-color: #1E2130; padding: 15px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# Initialisation des données
if "profile" not in st.session_state:
    st.session_state.profile = {"first_name": "", "last_name": "", "age": 0}
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {
        "balance_usd": 10000.0, 
        "assets": {"BTC": 0.0, "ETH": 0.0}, 
        "history": []  # Pour graphique d'évolution
    }

# Données crypto
@st.cache_data(ttl=300)
def get_crypto_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=5&page=1"
        return pd.DataFrame(requests.get(url, timeout=10).json())
    except:
        return pd.DataFrame([
            {"symbol": "BTC", "name": "Bitcoin", "current_price": 60000, "price_change_percentage_24h": 2.5},
            {"symbol": "ETH", "name": "Ethereum", "current_price": 4000, "price_change_percentage_24h": -1.2}
        ])

# Sauvegarde et chargement des données
def save_data():
    data = {"profile": st.session_state.profile, "portfolio": st.session_state.portfolio}
    st.download_button("Sauvegarder", json.dumps(data), "crypto_data.json", "application/json")

def load_data():
    uploaded_file = st.file_uploader("Charger vos données", type="json")
    if uploaded_file:
        data = json.load(uploaded_file)
        st.session_state.profile = data["profile"]
        st.session_state.portfolio = data["portfolio"]
        st.success("Données chargées !")

# Section Profil
def profile_section():
    st.subheader("👤 Votre Profil")
    st.session_state.profile["first_name"] = st.text_input("Prénom", st.session_state.profile["first_name"])
    st.session_state.profile["last_name"] = st.text_input("Nom", st.session_state.profile["last_name"])
    st.session_state.profile["age"] = st.number_input("Âge", 0, 150, st.session_state.profile["age"])
    if st.button("Enregistrer Profil"):
        st.success(f"Profil enregistré pour {st.session_state.profile['first_name']} {st.session_state.profile['last_name']} !")

# Section Portefeuille
def portfolio_section(df):
    st.subheader("👛 Portefeuille")
    total = st.session_state.portfolio["balance_usd"] + sum(
        qty * df[df["symbol"] == coin.lower()]["current_price"].iloc[0] 
        for coin, qty in st.session_state.portfolio["assets"].items()
    )
    profit = total - 10000  # Capital initial
    col1, col2 = st.columns(2)
    with col1: st.metric("Valeur totale", f"${total:,.2f}")
    with col2: st.metric("Profit/Perte", f"${profit:,.2f}", f"{(profit/10000*100):.2f}%")
    
    for coin, qty in st.session_state.portfolio["assets"].items():
        if qty > 0:
            st.write(f"{coin}: {qty:.4f}")
    
    # Graphique d'évolution
    if st.session_state.portfolio["history"]:
        history_df = pd.DataFrame(st.session_state.portfolio["history"])
        history_df["value"] = history_df.apply(
            lambda row: row["amount"] * row["price"] if row["type"] == "BUY" else -row["amount"] * row["price"], axis=1
        )
        fig = px.line(history_df, x="date", y="value", title="Évolution de vos Transactions")
        st.plotly_chart(fig)

# Section Budget
def budget_section():
    st.subheader("💵 Plan de Budget")
    st.markdown('<div class="budget-box">', unsafe_allow_html=True)
    income = st.number_input("Revenu mensuel ($)", 0.0, 100000.0, 2000.0)
    expenses = {
        "Logement": st.number_input("Logement", 0.0, income, 800.0),
        "Nourriture": st.number_input("Nourriture", 0.0, income, 300.0),
        "Transport": st.number_input("Transport", 0.0, income, 200.0),
        "Loisirs": st.number_input("Loisirs", 0.0, income, 150.0),
        "Autres": st.number_input("Autres", 0.0, income, 100.0)
    }
    total_expenses = sum(expenses.values())
    savings = income - total_expenses
    investable = max(0, savings * 0.5)
    
    st.write(f"Dépenses totales : ${total_expenses:,.2f}")
    st.write(f"Économies : ${savings:,.2f}")
    st.write(f"Montant investissable : ${investable:,.2f}")
    
    fig = px.pie(values=list(expenses.values()) + [savings], names=list(expenses.keys()) + ["Économies"], 
                 title="Répartition Budget")
    st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Section Analyse
def analysis_section(df):
    st.subheader("📊 Analyse")
    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
    st.write("### Marché")
    avg_change = df["price_change_percentage_24h"].mean()
    top_gainer = df.loc[df["price_change_percentage_24h"].idxmax()]
    st.write(f"Changement moyen 24h : {'+' if avg_change > 0 else ''}{avg_change:.2f}%")
    st.write(f"Meilleure crypto : {top_gainer['symbol'].upper()} (+{top_gainer['price_change_percentage_24h']:.2f}%)")
    fig = px.bar(df, x="symbol", y="price_change_percentage_24h", title="Performance 24h")
    st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Section Simulateur
def simulator():
    st.subheader("🎯 Simulateur")
    amount = st.number_input("Investissement ($)", 0.0, 10000.0)
    change = st.slider("Changement (%)", -50.0, 50.0, 0.0)
    result = amount * (1 + change/100)
    st.write(f"Résultat : ${result:,.2f}")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=result, title={"text": "Résultat ($)"}))
    st.plotly_chart(fig)

# Section Trading
def trading_section(df):
    st.subheader("💰 Trading")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="trade-box">', unsafe_allow_html=True)
        coin = st.selectbox("Crypto à acheter", df["symbol"].str.upper())
        amount = st.number_input("Montant ($)", 0.0, 10000.0, 100.0)
        if st.button("Acheter"):
            price = df[df["symbol"] == coin.lower()]["current_price"].iloc[0]
            qty = amount / price
            if amount <= st.session_state.portfolio["balance_usd"]:
                st.session_state.portfolio["balance_usd"] -= amount
                st.session_state.portfolio["assets"][coin] += qty
                st.session_state.portfolio["history"].append({
                    "type": "BUY", "symbol": coin, "amount": qty, "price": price, "date": datetime.now().strftime('%Y-%m-%d %H:%M')
                })
                st.success(f"🚀 Acheté {qty:.4f} {coin} !")
            else:
                st.error("Pas assez d’argent !")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="trade-box">', unsafe_allow_html=True)
        sell_coin = st.selectbox("Crypto à vendre", list(st.session_state.portfolio["assets"].keys()))
        qty = st.number_input("Quantité", 0.0, st.session_state.portfolio["assets"][sell_coin])
        if st.button("Vendre"):
            price = df[df["symbol"] == sell_coin.lower()]["current_price"].iloc[0]
            proceeds = qty * price
            st.session_state.portfolio["balance_usd"] += proceeds
            st.session_state.portfolio["assets"][sell_coin] -= qty
            st.session_state.portfolio["history"].append({
                "type": "SELL", "symbol": sell_coin, "amount": qty, "price": price, "date": datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            st.success(f"💸 Vendu pour ${proceeds:,.2f} !")
        st.markdown('</div>', unsafe_allow_html=True)

# Section Conseil
def advice_section(df):
    st.subheader("🧠 Conseils d’Investissement")
    for _, row in df.iterrows():
        symbol = row["symbol"].upper()
        change_24h = row["price_change_percentage_24h"]
        st.write(f"### {symbol}")
        if change_24h > 2:
            st.success(f"👍 {symbol} monte bien (+{change_24h:.2f}%). Bon moment pour acheter ou conserver !")
        elif change_24h < -2:
            st.warning(f"👇 {symbol} baisse (-{change_24h:.2f}%). Attendez ou vendez si vous avez.")
        else:
            st.info(f"⚖️ {symbol} est stable ({change_24h:.2f}%). Observez avant d’agir.")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=change_24h, 
                                    title={"text": f"Tendance 24h {symbol}"}, 
                                    gauge={"axis": {"range": [-10, 10]}}))
        st.plotly_chart(fig)

# Main
def main():
    st.title("Crypto Facile - Gérez Vos Cryptos !")
    df = get_crypto_data()
    
    # Sauvegarde et Chargement
    col1, col2 = st.columns(2)
    with col1: save_data()
    with col2: load_data()
    
    # Navigation
    page = st.radio("Menu", ["Profil", "Portefeuille", "Budget", "Analyse", "Simulateur", "Trading", "Conseil"], horizontal=True)
    
    if page == "Profil":
        profile_section()
    elif page == "Portefeuille":
        portfolio_section(df)
    elif page == "Budget":
        budget_section()
    elif page == "Analyse":
        analysis_section(df)
    elif page == "Simulateur":
        simulator()
    elif page == "Trading":
        trading_section(df)
    elif page == "Conseil":
        advice_section(df)

if __name__ == "__main__":
    main()
