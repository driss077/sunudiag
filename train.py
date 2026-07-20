"""SunuDiag - Entrainement du modele paludisme (DataSANTE-221).

Reprend le modele du cours d'Introduction au ML :
RandomForest sur (age, glycemie, hemoglobine, fievre, saison).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def generate_datasante221(n=10_000, seed=221):
    """Generateur DataSANTE-221 -- identique au cours de ML."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(2.5, 14, n), 5, 85).round(0).astype(int)
    glycemie = np.clip(rng.normal(5.5 + 0.05 * age, 1.8), 3.0, 18.0).round(1)
    hemoglobine = np.clip(rng.normal(12.5 - 0.02 * age, 1.5), 6.0, 17.0).round(1)
    fievre = np.clip(rng.normal(37.5, 0.9, n), 36.0, 41.5).round(1)
    saison = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    proba_palu = 1 / (1 + np.exp(-(-3 + 1.5 * saison + 0.8 * (fievre > 38.5))))
    palu = (rng.uniform(0, 1, n) < proba_palu).astype(int)
    return pd.DataFrame({"age": age, "glycemie": glycemie,
                         "hemoglobine": hemoglobine, "fievre": fievre,
                         "saison": saison, "paludisme": palu})


FEATURES = ["age", "glycemie", "hemoglobine", "fievre", "saison"]

if __name__ == "__main__":
    df = generate_datasante221()
    df.to_csv("data/datasante221.csv", index=False)

    X, y = df[FEATURES], df["paludisme"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)

    auc = roc_auc_score(y_test, modele.predict_proba(X_test)[:, 1])
    print(f"AUC test : {auc:.3f}")

    joblib.dump(modele, "models/model.pkl")
    print("Modele sauvegarde dans models/model.pkl")