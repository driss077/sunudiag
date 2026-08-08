# L'image de depart : un Linux minimal avec Python 3.11
FROM python:3.11-slim

# Le repertoire de travail a l'interieur du conteneur
WORKDIR /app

# Installer les dependances AVANT de copier le code (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet : API, frontend et modele
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Port 10000 pour Render.com
EXPOSE 10000

# La commande lancee au demarrage du conteneur
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]