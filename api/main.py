"""SunuDiag - API REST servant le modele paludisme (Lab 2)."""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SunuDiag API",
    description=(
        "Pre-diagnostic du paludisme (modele DataSANTE-221). "
        "Un pre-diagnostic n’est pas un diagnostic medical."
    ),
    version="2.0",
)
# Autoriser le navigateur a appeler l'API depuis une autre origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Le modele est charge UNE SEULE FOIS, au demarrage du serveur
modele = joblib.load("models/model.pkl")
FEATURES = ["age", "glycemie", "hemoglobine", "fievre", "saison"]


class Patient(BaseModel):
    """Le contrat d’interface : ce que l’API accepte en entree."""

    age: int = Field(ge=0, le=120, description="Age en annees")
    glycemie: float = Field(ge=2.0, le=25.0, description="Glycemie (mmol/L)")
    hemoglobine: float = Field(ge=4.0, le=20.0, description="Hemoglobine (g/dL)")
    fievre: float = Field(ge=34.0, le=43.0, description="Temperature (C)")
    saison: int = Field(ge=0, le=1, description="1 = saison des pluies")


@app.get("/health")
def health():
    """Verifier que l’API est en vie et que le modele est charge."""
    return {"statut": "ok", "modele": "RandomForest DataSANTE-221"}


@app.post("/predict")
def predict(patient: Patient):
    """Predire le risque de paludisme pour un patient."""
    donnees = pd.DataFrame([patient.model_dump()])[FEATURES]
    proba = float(modele.predict_proba(donnees)[0, 1])
    return {
        "probabilite_paludisme": round(proba, 3),
        "pre_diagnostic": "A ORIENTER" if proba >= 0.5 else "risque faible",
        "avertissement": "Ne remplace pas un avis medical.",
    }

