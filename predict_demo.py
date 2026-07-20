"""
SunuDiag - Demonstration : recharger le modele et predire.
"""
import pandas as pd
import joblib

modele = joblib.load("models/model.pkl")

# Moussa, 34 ans, se presente au poste de sante de Thies :
# fievre a 39.2 C, en saison des pluies.
patient = pd.DataFrame([{
    "age": 34,
    "glycemie": 5.8,
    "hemoglobine": 13.1,
    "fievre": 39.2,
    "saison": 1,
}])

proba = modele.predict_proba(patient)[0, 1]
print(f"Probabilite de paludisme : {proba:.1%}")
print("Pre-diagnostic :", "A ORIENTER" if proba >= 0.5 else "risque faible")