import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
from textblob import TextBlob
import time
import random
from newsapi import NewsApiClient
from prophet import Prophet
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Chargez les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(page_title="Tableau de Bord Financier", layout="wide")

# Fonctions financières de base
def calcul_interets_composes(principal, taux, annees):
    return principal * (1 + taux / 100) ** annees

def calcul_amortissement_pret(principal, taux_annuel, annees):
    taux_mensuel = taux_annuel / 12 / 100
    nombre_paiements = annees * 12
    paiement_mensuel = principal * (taux_mensuel * (1 + taux_mensuel) ** nombre_paiements) / ((1 + taux_mensuel) ** nombre_paiements - 1)
    return paiement_mensuel

def calcul_van(flux, taux):
    return sum(f / (1 + taux / 100) ** (t + 1) for t, f in enumerate(flux))

def calcul_tri(flux):
    def van_at_rate(rate):
        return sum(f / (1 + rate) ** (t + 1) for t, f in enumerate(flux))
    low, high = -0.99, 100.0
    for _ in range(100):
        mid = (low + high) / 2
        van = van_at_rate(mid)
        if abs(van) < 0.01:
            return mid * 100
        elif van > 0:
            low = mid
        else:
            high = mid
    return mid * 100

@st.cache_data
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty:
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        else:
            st.warning(f"Aucune donnée trouvée pour {ticker} sur la période {period}.")
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {ticker} : {e}")
        return pd.DataFrame()

def predict_stock_price(df):
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

def monte_carlo_simulation(initial_investment, mean_return, volatility, years, simulations=1000):
    if simulations > 5000:
        st.warning("Nombre de simulations réduit à 5000 pour des raisons de performance.")
        simulations = 5000
    total_days = 252 * years
    daily_returns = np.random.normal(mean_return / 252, volatility / np.sqrt(252), (total_days, simulations))
    price_paths = initial_investment * np.exp(np.cumsum(daily_returns, axis=0))
    return price_paths

def get_stock_params(ticker, period="1y"):
    df = get_stock_data(ticker, period)
    if df.empty:
        return 0, 0
    daily_returns = df['Close'].pct_change().dropna()
    mean_return = daily_returns.mean() * 252
    volatility = daily_returns.std() * np.sqrt(252)
    return mean_return, volatility

def generate_enriched_text(prompt, data, key):
    base_text = f"{key} : {data:.2f}"
    blob = TextBlob(prompt)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return f"{base_text}. Cette valeur reflète une tendance positive, suggérant une opportunité favorable."
    elif sentiment < 0:
        return f"{base_text}. Cette valeur indique une situation préoccupante qui mérite une attention particulière."
    else:
        return f"{base_text}. Les données sont neutres, sans tendance marquée."

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def generate_pdf_report(selected_data, filename=None):
    if filename is None:
        fd, filename = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    c.drawString(100, y, "Rapport Financier Personnalisé")
    y -= 20
    for key, value in selected_data.items():
        if isinstance(value, dict):
            c.drawString(100, y, f"{key} :")
            y -= 20
            for sub_key, sub_value in value.items():
                c.drawString(120, y, f"- {sub_key} : {sub_value}")
                y -= 20
        else:
            c.drawString(100, y, f"{key} : {value}")
            y -= 20
            if y < 50:
                c.showPage()
                y = 750
    c.save()
    return filename

# Questions pour le quiz (exemple réduit)
easy_questions = [
    {"question": "Que sont les intérêts composés ?", "options": ["Intérêts sur le capital initial", "Intérêts sur le capital + intérêts cumulés", "Intérêts fixes"], "correct": "Intérêts sur le capital + intérêts cumulés", "explanation": "Les intérêts composés génèrent des intérêts sur le capital initial et sur les intérêts déjà accumulés."},
    {"question": "Qu’est-ce qu’un dividende ?", "options": ["Un prêt bancaire", "Une part des bénéfices distribuée", "Un taux d’intérêt"], "correct": "Une part des bénéfices distribuée", "explanation": "Un dividende est une partie des profits qu’une entreprise partage avec ses actionnaires."},
]

medium_questions = [
    {"question": "Que mesure la volatilité ?", "options": ["Le rendement moyen", "L’écart des rendements", "Le prix d’une action"], "correct": "L’écart des rendements", "explanation": "La volatilité indique à quel point les rendements d’un actif fluctuent autour de leur moyenne."},
    {"question": "Qu’est-ce que la VAN ?", "options": ["La valeur future d’un investissement", "La valeur actuelle des flux de trésorerie", "Le taux de rendement"], "correct": "La valeur actuelle des flux de trésorerie", "explanation": "La VAN actualise les flux futurs pour estimer leur valeur aujourd’hui."},
]

hard_questions = [
    {"question": "Que représente le TRI ?", "options": ["Le taux d’actualisation rendant la VAN nulle", "Le rendement moyen d’un portefeuille", "Le taux d’intérêt d’un prêt"], "correct": "Le taux d’actualisation rendant la VAN nulle", "explanation": "Le TRI est le taux qui équilibre les entrées et sorties de trésorerie dans un projet."},
    {"question": "Quel indicateur mesure le risque ajusté au rendement ?", "options": ["Ratio Sharpe", "Beta", "Volatilité"], "correct": "Ratio Sharpe", "explanation": "Le ratio Sharpe évalue le rendement excédentaire par unité de risque."},
]

# Liste des sections disponibles
SECTIONS = [
    "Accueil (KPI)",
    "Calculatrices Financières",
    "Analyse de Portefeuille",
    "Visualisation Boursière",
    "Prédiction de Prix",
    "Simulation Monte Carlo",
    "Analyse de Sentiments",
    "Quiz Financier",
    "Rapport Personnalisé"
]

# Page de connexion
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_info = {}
    st.session_state.preferred_section = None

if not st.session_state.authenticated:
    st.title("Bienvenue sur le Tableau de Bord Financier")
    st.subheader("Veuillez vous authentifier")
    with st.form(key='login_form'):
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        age = st.number_input("Âge", min_value=1, max_value=120, step=1)
        email = st.text_input("Email (optionnel)", "")
        submit_button = st.form_submit_button(label="Se connecter")
        if submit_button and nom and prenom and age:
            st.session_state.user_info = {"nom": nom, "prenom": prenom, "age": age, "email": email if email else "Non fourni"}
            st.session_state.authenticated = True
            st.success(f"Connexion réussie, {prenom} !")
            st.rerun()
        else:
            st.error("Veuillez remplir tous les champs obligatoires (Nom, Prénom, Âge).")

elif st.session_state.authenticated and st.session_state.preferred_section is None:
    st.title(f"Bonjour, {st.session_state.user_info['prenom']} !")
    st.subheader("Que souhaitez-vous explorer aujourd’hui ?")
    preferred_section = st.selectbox("Choisissez une section", SECTIONS)
    if st.button("Confirmer"):
        st.session_state.preferred_section = preferred_section
        st.success(f"Vous avez choisi : {preferred_section}")
        st.rerun()

else:
    prenom = st.session_state.user_info['prenom']
    age = st.session_state.user_info['age']
    st.sidebar.title(f"Bienvenue, {prenom} !")
    st.sidebar.write(f"Âge : {age}")
    section = st.sidebar.radio("Choisir une section", SECTIONS, index=SECTIONS.index(st.session_state.preferred_section))
    
    if age < 30:
        st.sidebar.write("Astuce : Pensez à investir tôt pour profiter des intérêts composés !")
    elif age >= 50:
        st.sidebar.write("Astuce : Diversifiez pour réduire les risques à l’approche de la retraite.")

    # Section "Accueil (KPI)"
    if section == "Accueil (KPI)":
        st.title(f"Tableau de Bord Financier - {prenom}")
        st.subheader("Indicateurs Clés de Performance (KPI)")

        col_input1, col_input2 = st.columns(2)
        with col_input1:
            portfolio_tickers = st.text_input("Actifs du portefeuille (ex: AAPL, MSFT)", "AAPL, MSFT").split(",")
        with col_input2:
            portfolio_values = st.text_input("Montants investis (ex: 1000, 2000)", "1000, 2000").split(",")
        period = st.selectbox("Période des données", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

        if len(portfolio_tickers) != len(portfolio_values):
            st.error("Le nombre d'actifs et de montants doit correspondre !")
        else:
            portfolio_values_raw = [v.strip() for v in portfolio_values]
            portfolio_values_num = []
            all_valid = True
            for val in portfolio_values_raw:
                if val.replace(".", "").isdigit() or (val.startswith("-") and val[1:].replace(".", "").isdigit()):
                    portfolio_values_num.append(float(val))
                else:
                    all_valid = False
                    st.error(f"'{val}' n'est pas un nombre valide. Utilisez des nombres (ex: 1000, 2000).")
                    break

            if all_valid:
                total_value = sum(portfolio_values_num)
                with st.spinner("Calcul des indicateurs..."):
                    returns, volatilities = [], []
                    for ticker in portfolio_tickers:
                        mean_ret, vol = get_stock_params(ticker.strip(), period)
                        returns.append(mean_ret)
                        volatilities.append(vol)
                    avg_return = np.mean(returns) * 100
                    avg_volatility = np.mean(volatilities) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Valeur Totale", f"{total_value:.2f} €")
                col2.metric("Rendement Moyen Annualisé", f"{avg_return:.2f} %", delta=f"{avg_return:.2f}%", delta_color="normal")
                col3.metric("Volatilité Moyenne", f"{avg_volatility:.2f} %", delta_color="off")

                st.subheader("Aperçu des Prix")
                fig = px.line(title=f"Prix sur {period}")
                for ticker in portfolio_tickers:
                    df = get_stock_data(ticker.strip(), period)
                    if not df.empty:
                        fig.add_scatter(x=df.index, y=df["Close"], mode="lines", name=ticker)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Actualités Financières")
                api_key = os.getenv("NEWSAPI_KEY")
                if not api_key:
                    st.error("Clé API NewsAPI manquante. Veuillez configurer la variable d'environnement NEWSAPI_KEY.")
                else:
                    newsapi = NewsApiClient(api_key=api_key)
                    for ticker in portfolio_tickers:
                        with st.expander(f"Actualités pour {ticker.strip()} (jusqu'à 60 articles)", expanded=False):
                            try:
                                news = newsapi.get_everything(q=ticker.strip(), language='fr', sort_by='publishedAt', page_size=60)
                                articles = news['articles']
                                if articles:
                                    df_news = pd.DataFrame({
                                        "Titre": [article['title'] for article in articles],
                                        "Date": [article['publishedAt'][:10] for article in articles],
                                        "Source": [article['source']['name'] for article in articles],
                                        "Lien": [f"[Lire]({article['url']})" for article in articles]
                                    })
                                    st.dataframe(df_news, use_container_width=True)
                                else:
                                    st.write("Aucune actualité trouvée.")
                            except Exception as e:
                                st.error(f"Erreur pour {ticker} : {e}")

                    st.subheader("Recherche d'Actualités Personnalisée")
                    col_search1, col_search2, col_search3 = st.columns(3)
                    with col_search1:
                        search_query = st.text_input("Ticker ou mot-clé (ex: TSLA, Bitcoin)", "")
                    with col_search2:
                        date_from = st.date_input("À partir de", value=datetime.now() - timedelta(days=30))
                    with col_search3:
                        sources = st.multiselect("Sources (optionnel)", ["Le Monde", "Les Echos", "Reuters", "AFP"], default=[])
                    
                    if st.button("Rechercher") and search_query:
                        with st.spinner("Recherche en cours..."):
                            try:
                                source_str = ",".join([s.lower().replace(" ", "-") for s in sources]) if sources else None
                                news = newsapi.get_everything(
                                    q=search_query.strip(),
                                    language='fr',
                                    sort_by='publishedAt',
                                    page_size=60,
                                    from_param=date_from.strftime("%Y-%m-%d"),
                                    sources=source_str
                                )
                                articles = news['articles']
                                if articles:
                                    df_search = pd.DataFrame({
                                        "Titre": [article['title'] for article in articles],
                                        "Date": [article['publishedAt'][:10] for article in articles],
                                        "Source": [article['source']['name'] for article in articles],
                                        "Lien": [f"[Lire]({article['url']})" for article in articles]
                                    })
                                    st.write(f"Résultats pour '{search_query}' (jusqu'à 60 articles) :")
                                    st.dataframe(df_search, use_container_width=True)
                                else:
                                    st.write(f"Aucune actualité trouvée pour '{search_query}'.")
                            except Exception as e:
                                st.error(f"Erreur lors de la recherche : {e}")

    # Section "Calculatrices Financières"
    elif section == "Calculatrices Financières":
        st.title(f"Calculatrices Financières - {prenom}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Intérêts Composés")
            principal = st.number_input("Montant initial (€)", min_value=0.0, value=1000.0)
            taux = st.slider("Taux annuel (%)", 0.0, 20.0, 5.0)
            annees = st.slider("Durée (années)", 1, 30, 10)
            resultat = calcul_interets_composes(principal, taux, annees)
            st.write(f"Valeur future : **{resultat:.2f} €**")
        with col2:
            st.subheader("Amortissement de Prêt")
            pret_principal = st.number_input("Montant du prêt (€)", min_value=0.0, value=50000.0)
            pret_taux = st.slider("Taux annuel (%) ", 0.0, 15.0, 3.0)
            pret_annees = st.slider("Durée (années) ", 1, 30, 15)
            paiement = calcul_amortissement_pret(pret_principal, pret_taux, pret_annees)
            st.write(f"Paiement mensuel : **{paiement:.2f} €**")
        with col3:
            st.subheader("VAN et TRI")
            flux_input = st.text_area("Flux (ex: -5000, 2000, 3000)", "-5000, 2000, 3000")
            flux_raw = [x.strip() for x in flux_input.split(",")]
            flux = []
            all_valid = True
            for f in flux_raw:
                if f.replace(".", "").isdigit() or (f.startswith("-") and f[1:].replace(".", "").isdigit()):
                    flux.append(float(f))
                else:
                    all_valid = False
                    st.error(f"'{f}' n'est pas un nombre valide.")
                    break
            if all_valid:
                van_taux = st.slider("Taux d'actualisation (%)", 0.0, 20.0, 5.0)
                van = calcul_van(flux, van_taux)
                tri = calcul_tri(flux)
                st.write(f"VAN : **{van:.2f} €**")
                st.write(f"TRI : **{tri:.2f} %**")

    # Section "Analyse de Portefeuille"
    elif section == "Analyse de Portefeuille":
        st.title(f"Analyse de Portefeuille - {prenom}")
        
        # Entrées utilisateur
        st.subheader("Composition du portefeuille")
        col1, col2 = st.columns(2)
        with col1:
            actifs = st.text_input("Actifs (ex: AAPL, MSFT)", "AAPL, MSFT")
        with col2:
            montants = st.text_input("Montants investis (ex: 1000, 2000)", "1000, 2000")
        period = st.selectbox("Période d'analyse", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

        actifs_list = [x.strip().upper() for x in actifs.split(",")]
        montants_list_raw = [x.strip() for x in montants.split(",")]

        # Validation des montants
        montants_list = []
        all_valid = True
        for montant in montants_list_raw:
            if montant.replace(".", "").isdigit() or (montant.startswith("-") and montant[1:].replace(".", "").isdigit()):
                montants_list.append(float(montant))
            else:
                all_valid = False
                st.error(f"'{montant}' n'est pas un nombre valide. Utilisez des nombres (ex: 1000, 2000).")
                break

        if all_valid and len(actifs_list) == len(montants_list):
            # Création du DataFrame initial
            df_portfolio = pd.DataFrame({"Actif": actifs_list, "Montant Initial": montants_list})

            # 1. Valeur actuelle et performance
            st.subheader("Valeur Actuelle et Performance")
            current_values = []
            historical_data = {}
            for ticker in actifs_list:
                df = get_stock_data(ticker, period)
                if not df.empty:
                    current_price = df['Close'][-1]
                    initial_amount = df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0]
                    current_value = (initial_amount / df['Close'][0]) * current_price
                    current_values.append(current_value)
                    historical_data[ticker] = df['Close']
                else:
                    current_values.append(df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0])
                    st.warning(f"Données indisponibles pour {ticker}. Valeur initiale utilisée.")
            df_portfolio["Valeur Actuelle"] = current_values
            total_initial = df_portfolio["Montant Initial"].sum()
            total_current = df_portfolio["Valeur Actuelle"].sum()
            rendement_total = ((total_current - total_initial) / total_initial) * 100
            rendement_annualise = ((total_current / total_initial) ** (1 / (int(period[:-2]) / 12)) - 1) * 100 if period.endswith("mo") else ((total_current / total_initial) ** (1 / int(period[:-1])) - 1) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Valeur Initiale", f"{total_initial:.2f} €")
            col2.metric("Valeur Actuelle", f"{total_current:.2f} €")
            col3.metric("Rendement Total", f"{rendement_total:.2f} %", delta=f"{rendement_annualise:.2f}% annualisé")

            # Affichage du tableau
            st.dataframe(df_portfolio, use_container_width=True)

            # 2. Répartition par secteur
            st.subheader("Répartition par Secteur")
            sectors = {}
            for ticker in actifs_list:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get("sector", "Inconnu")
                sectors[ticker] = sector
            df_portfolio["Secteur"] = [sectors[ticker] for ticker in df_portfolio["Actif"]]
            sector_dist = df_portfolio.groupby("Secteur")["Valeur Actuelle"].sum().reset_index()
            fig_sector = px.pie(sector_dist, values="Valeur Actuelle", names="Secteur", title="Répartition par Secteur")
            st.plotly_chart(fig_sector, use_container_width=True)

            # 3. Évolution temporelle
            st.subheader("Évolution Temporelle")
            df_historical = pd.DataFrame(historical_data)
            df_historical.index = df_historical.index.tz_localize(None)
            portfolio_value = pd.DataFrame(index=df_historical.index)
            for ticker in actifs_list:
                initial_amount = df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0]
                shares = initial_amount / df_historical[ticker][0]
                portfolio_value[ticker] = df_historical[ticker] * shares
            portfolio_value["Total"] = portfolio_value.sum(axis=1)
            fig_time = px.line(portfolio_value, x=portfolio_value.index, y="Total", title=f"Évolution du Portefeuille ({period})")
            st.plotly_chart(fig_time, use_container_width=True)

            # 4. Volatilité du portefeuille
            st.subheader("Risque du Portefeuille")
            returns = df_historical.pct_change().dropna()
            portfolio_returns = (returns * (df_portfolio["Montant Initial"] / total_initial).values).sum(axis=1)
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            st.metric("Volatilité Annualisée", f"{volatility:.2f} %")

            # 5. Matrice de corrélation
            st.subheader("Corrélation entre Actifs")
            correlation_matrix = returns.corr()
            fig_corr = px.imshow(correlation_matrix, text_auto=True, title="Matrice de Corrélation", color_continuous_scale="RdBu")
            st.plotly_chart(fig_corr, use_container_width=True)

            # 6. Répartition initiale
            st.subheader("Répartition Initiale")
            fig_pie = px.pie(df_portfolio, values="Montant Initial", names="Actif", title="Répartition Initiale")
            st.plotly_chart(fig_pie, use_container_width=True)
        elif all_valid:
            st.error("Le nombre d'actifs et de montants doit correspondre !")

    # Section "Visualisation Boursière"
    elif section == "Visualisation Boursière":
        st.title(f"Visualisation Boursière - {prenom}")
        tickers = st.text_input("Symboles (ex: AAPL, MSFT)", "AAPL, MSFT").split(",")
        period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y"])
        with st.spinner("Chargement..."):
            fig = px.line(title="Prix de clôture")
            for ticker in tickers:
                df = get_stock_data(ticker.strip(), period)
                if not df.empty:
                    fig.add_scatter(x=df.index, y=df["Close"], mode="lines", name=ticker)
            st.plotly_chart(fig)

    # Section "Prédiction de Prix"
    elif section == "Prédiction de Prix":
        st.title(f"Prédiction de Prix - {prenom}")
        ticker_pred = st.text_input("Symbole (ex: AAPL)", "AAPL")
        model_choice = st.selectbox("Modèle", ["Régression Linéaire", "Prophet"])
        df_pred = get_stock_data(ticker_pred)
        if not df_pred.empty:
            with st.spinner("Calcul..."):
                if model_choice == "Régression Linéaire":
                    df_pred['Days'] = np.arange(len(df_pred))
                    X = df_pred[['Days']]
                    y = df_pred['Close']
                    model = LinearRegression()
                    model.fit(X, y)
                    future_days = np.arange(len(df_pred), len(df_pred) + 30).reshape(-1, 1)
                    predictions = model.predict(future_days)
                    dates_future = pd.date_range(start=df_pred.index[-1], periods=31, freq='B')[1:]
                    explanation = "Tendance linéaire simple."
                else:
                    df_prophet = df_pred.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                    model = Prophet(daily_seasonality=True)
                    model.fit(df_prophet)
                    future = model.make_future_dataframe(periods=30, freq='B')
                    forecast = model.predict(future)
                    predictions = forecast['yhat'].tail(30)
                    dates_future = forecast['ds'].tail(30)
                    explanation = "Tendances et saisonnalités."
                fig_pred = px.line(title=f"Prédiction {ticker_pred} ({model_choice})")
                fig_pred.add_scatter(x=df_pred.index, y=df_pred['Close'], mode='lines', name='Historique')
                fig_pred.add_scatter(x=dates_future, y=predictions, mode='lines', name='Prédiction', line=dict(dash='dash'))
                st.plotly_chart(fig_pred)
                st.write(f"**Explication** : {explanation}")

    # Section "Simulation Monte Carlo"
    elif section == "Simulation Monte Carlo":
        st.title(f"Simulation Monte Carlo - {prenom}")
        invest = st.number_input("Investissement (€)", min_value=0.0, value=10000.0)
        sim_years = st.slider("Durée (années)", 1, 20, 5)
        num_traj = st.slider("Trajectoires", 1, 50, 10)
        mode = st.radio("Mode", ["Basé sur un titre", "Manuel"])
        if mode == "Basé sur un titre":
            ticker = st.text_input("Symbole (ex: AAPL)", "AAPL")
            period = st.selectbox("Période", ["1y", "2y", "5y"])
            mean_ret, vol = get_stock_params(ticker, period)
        else:
            mean_ret = st.slider("Rendement (%)", 0.0, 20.0, 5.0) / 100
            vol = st.slider("Volatilité (%)", 0.0, 50.0, 15.0) / 100
        paths = monte_carlo_simulation(invest, mean_ret, vol, sim_years)
        df_paths = pd.DataFrame(paths, columns=[f"Sim {i+1}" for i in range(paths.shape[1])])
        df_paths['Days'] = np.arange(252 * sim_years)
        df_paths['Mean'] = df_paths.iloc[:, :-1].mean(axis=1)
        cols_to_plot = df_paths.columns[:num_traj].tolist() + ['Mean']
        fig_mc = px.line(df_paths, x="Days", y=cols_to_plot, title="Simulation")
        st.plotly_chart(fig_mc)

    # Section "Analyse de Sentiments"
    elif section == "Analyse de Sentiments":
        st.title(f"Analyse de Sentiments - {prenom}")
        texte = st.text_area("Texte", "Le marché est en hausse !")
        sentiment = analyze_sentiment(texte)
        st.write(f"Sentiment : **{'Positif' if sentiment > 0 else 'Négatif' if sentiment < 0 else 'Neutre'}** (Score: {sentiment:.2f})")

    # Section "Quiz Financier"
    elif section == "Quiz Financier":
        st.title(f"Quiz Financier - {prenom}")
        difficulty = st.selectbox("Niveau", ["Facile", "Moyen", "Difficile"])
        question_sets = {"Facile": easy_questions, "Moyen": medium_questions, "Difficile": hard_questions}
        selected_questions = question_sets[difficulty]
        if 'quiz_questions' not in st.session_state or st.session_state.quiz_difficulty != difficulty:
            st.session_state.quiz_questions = random.sample(selected_questions, min(10, len(selected_questions)))
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.session_state.quiz_finished = False
            st.session_state.timer_start = None
            st.session_state.quiz_difficulty = difficulty
        if not st.session_state.quiz_finished:
            q_index = st.session_state.current_question
            question_data = st.session_state.quiz_questions[q_index]
            st.subheader(f"Question {q_index + 1}/10")
            st.write(question_data["question"])
            if st.session_state.timer_start is None:
                st.session_state.timer_start = time.time()
            time_left = max(0, 30 - (time.time() - st.session_state.timer_start))
            st.write(f"Temps restant : {int(time_left)}s")
            user_answer = st.radio("Réponse", question_data["options"], key=f"q{q_index}")
            if st.button("Soumettre", key=f"submit{q_index}") or time_left <= 0:
                if time_left <= 0:
                    st.error("Temps écoulé !")
                elif user_answer == question_data["correct"]:
                    st.success("Correct !")
                    st.session_state.score += 1
                else:
                    st.error(f"Faux. Réponse : {question_data['correct']}")
                st.write(f"Explication : {question_data['explanation']}")
                st.session_state.current_question += 1
                st.session_state.timer_start = None
                if st.session_state.current_question >= 10:
                    st.session_state.quiz_finished = True
                st.rerun()
        if st.session_state.quiz_finished:
            st.subheader("Quiz Terminé !")
            score = st.session_state.score
            st.write(f"Votre score : **{score}/10** ({score * 10}%)")
            if st.button("Recommencer"):
                del st.session_state.quiz_questions
                del st.session_state.current_question
                del st.session_state.score
                del st.session_state.quiz_finished
                del st.session_state.timer_start
                del st.session_state.quiz_difficulty
                st.rerun()

    # Section "Rapport Personnalisé"
    elif section == "Rapport Personnalisé":
        st.title(f"Générateur de Rapport Personnalisé - {prenom}")
        st.subheader("Que voulez-vous inclure dans votre rapport ?")
        
        inclure_interets = st.checkbox("Intérêts Composés")
        inclure_pret = st.checkbox("Amortissement de Prêt")
        inclure_van = st.checkbox("Valeur Actuelle Nette (VAN)")
        inclure_tri = st.checkbox("Taux de Rentabilité Interne (TRI)")
        inclure_kpi = st.checkbox("Indicateurs Clés (KPI)")
        inclure_sentiment = st.checkbox("Analyse de Sentiment")
        inclure_graphique = st.checkbox("Graphique Boursier")
        inclure_monte_carlo = st.checkbox("Résultats Monte Carlo")

        selected_data = {}
        
        if inclure_interets:
            principal = st.number_input("Montant initial (€)", min_value=0.0, value=1000.0, key="interets_principal")
            taux = st.slider("Taux annuel (%)", 0.0, 20.0, 5.0, key="interets_taux")
            annees = st.slider("Durée (années)", 1, 30, 10, key="interets_annees")
            resultat = calcul_interets_composes(principal, taux, annees)
            selected_data["Intérêts Composés"] = generate_enriched_text("Valeur future des intérêts", resultat, "Montant final")

        if inclure_pret:
            pret_principal = st.number_input("Montant du prêt (€)", min_value=0.0, value=50000.0, key="pret_principal")
            pret_taux = st.slider("Taux annuel (%)", 0.0, 15.0, 3.0, key="pret_taux")
            pret_annees = st.slider("Durée (années)", 1, 30, 15, key="pret_annees")
            paiement = calcul_amortissement_pret(pret_principal, pret_taux, pret_annees)
            selected_data["Amortissement de Prêt"] = generate_enriched_text("Paiement mensuel du prêt", paiement, "Paiement mensuel")

        if inclure_van or inclure_tri:
            flux_input = st.text_input("Flux de trésorerie (ex: -5000, 2000, 3000)", "-5000, 2000, 3000", key="flux_input")
            flux_raw = [x.strip() for x in flux_input.split(",")]
            flux = []
            all_valid = True
            for f in flux_raw:
                if f.replace(".", "").isdigit() or (f.startswith("-") and f[1:].replace(".", "").isdigit()):
                    flux.append(float(f))
                else:
                    all_valid = False
                    st.error(f"'{f}' n'est pas un nombre valide.")
                    break
            if all_valid:
                if inclure_van:
                    van_taux = st.slider("Taux d'actualisation (%)", 0.0, 20.0, 5.0, key="van_taux")
                    van = calcul_van(flux, van_taux)
                    selected_data["VAN"] = generate_enriched_text("Valeur actuelle nette", van, "VAN")
                if inclure_tri:
                    tri = calcul_tri(flux)
                    selected_data["TRI"] = generate_enriched_text("Taux de rentabilité interne", tri, "TRI")

        if inclure_kpi:
            portfolio_tickers = st.text_input("Actifs (ex: AAPL, MSFT)", "AAPL, MSFT", key="kpi_tickers").split(",")
            portfolio_values = st.text_input("Montants (ex: 1000, 2000)", "1000, 2000", key="kpi_values").split(",")
            if len(portfolio_tickers) == len(portfolio_values):
                portfolio_values_raw = [v.strip() for v in portfolio_values]
                portfolio_values_num = []
                all_valid = True
                for val in portfolio_values_raw:
                    if val.replace(".", "").isdigit() or (val.startswith("-") and val[1:].replace(".", "").isdigit()):
                        portfolio_values_num.append(float(val))
                    else:
                        all_valid = False
                        st.error(f"'{val}' n'est pas un nombre valide.")
                        break
                if all_valid:
                    total_value = sum(portfolio_values_num)
                    returns, volatilities = [], []
                    for ticker in portfolio_tickers:
                        mean_ret, vol = get_stock_params(ticker.strip())
                        returns.append(mean_ret)
                        volatilities.append(vol)
                    avg_return = np.mean(returns) * 100
                    avg_volatility = np.mean(volatilities) * 100
                    selected_data["KPI"] = {
                        "Valeur Totale": f"{total_value:.2f} €",
                        "Rendement Moyen": f"{avg_return:.2f} %",
                        "Volatilité Moyenne": f"{avg_volatility:.2f} %"
                    }
            else:
                st.error("Le nombre d'actifs et de montants doit correspondre !")

        if inclure_sentiment:
            texte = st.text_area("Texte pour analyse", "Le marché est en hausse !", key="sentiment_texte")
            sentiment = analyze_sentiment(texte)
            selected_data["Sentiment"] = f"Score : {sentiment:.2f} ({'Positif' if sentiment > 0 else 'Négatif' if sentiment < 0 else 'Neutre'})"

        if inclure_graphique:
            ticker = st.text_input("Symbole (ex: AAPL)", "AAPL", key="graph_ticker")
            period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y"], key="graph_period")
            df = get_stock_data(ticker, period)
            if not df.empty:
                fig = px.line(df, x=df.index, y="Close", title=f"Performance de {ticker}")
                st.plotly_chart(fig)
                selected_data["Graphique"] = f"Graphique de {ticker} pour {period} inclus."

        if inclure_monte_carlo:
            invest = st.number_input("Investissement (€)", min_value=0.0, value=10000.0, key="mc_invest")
            sim_years = st.slider("Durée (années)", 1, 20, 5, key="mc_years")
            mean_ret = st.slider("Rendement (%)", 0.0, 20.0, 5.0, key="mc_ret") / 100
            vol = st.slider("Volatilité (%)", 0.0, 50.0, 15.0, key="mc_vol") / 100
            paths = monte_carlo_simulation(invest, mean_ret, vol, sim_years)
            final_values = paths[-1]
            selected_data["Monte Carlo"] = {
                "Moyenne": f"{np.mean(final_values):.2f} €",
                "5e Percentile": f"{np.percentile(final_values, 5):.2f} €",
                "95e Percentile": f"{np.percentile(final_values, 95):.2f} €"
            }

        if st.button("Générer le Rapport"):
            if selected_data:
                pdf_path = generate_pdf_report(selected_data)
                with open(pdf_path, "rb") as file:
                    st.download_button("Télécharger le Rapport PDF", file, "rapport_personnalise.pdf")
                os.remove(pdf_path)
            else:
                st.warning("Veuillez sélectionner au moins une option.")

# Footer
if st.session_state.authenticated:
    st.sidebar.write(f"Session de {st.session_state.user_info['prenom']} - Mars 2025")
