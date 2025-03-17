print("mon premier fichier python")
import streamlit as st

# Titre de l'application
st.title("Calculateur d'Intérêts Composés")

# Description
st.write("Calculez la croissance de votre investissement avec les intérêts composés.")

# Entrées utilisateur
capital = st.number_input("Capital initial (€)", min_value=0.0, value=1000.0, step=100.0)
taux = st.number_input("Taux d'intérêt annuel (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
annees = st.number_input("Durée (années)", min_value=1, max_value=100, value=10, step=1)

# Calcul des intérêts composés
if st.button("Calculer"):
    montant_final = capital * (1 + taux/100) ** annees
    interets = montant_final - capital
    
    # Affichage des résultats
    st.subheader("Résultats")
    st.write(f"Capital initial : {capital:,.2f} €")
    st.write(f"Taux d'intérêt : {taux}% par an")
    st.write(f"Durée : {annees} ans")
    st.write(f"Montant final : {montant_final:,.2f} €")
    st.write(f"Intérêts gagnés : {interets:,.2f} €")

# Instructions pour lancer l'app
st.write("*Pour lancer cette application, sauvegardez le code dans un fichier (ex: app.py) et exécutez `streamlit run app.py` dans votre terminal.*")