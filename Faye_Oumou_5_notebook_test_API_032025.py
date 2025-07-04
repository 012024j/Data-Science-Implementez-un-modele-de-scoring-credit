# Importation des librairies nécessaires
from fastapi import FastAPI, Request  # Framework pour créer des APIs rapides
from pydantic import BaseModel  # Permet de définir et valider les modèles de données d'entrée
from fastapi.middleware.cors import CORSMiddleware  # Middleware pour gérer les politiques CORS
from xgboost import XGBClassifier  # Importation du modèle XGBoost
import numpy as np  # Manipulation de tableaux et calcul scientifique
import joblib  # Pour charger et sauvegarder des modèles ML
import uvicorn  # Serveur ASGI pour lancer l'app FastAPI
import os  # Gestion des chemins et opérations sur le système de fichiers

# Initialisation de l'application FastAPI
app = FastAPI()

# Ajout du middleware pour gérer CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP
    allow_headers=["*"],  # Autorise tous les headers
)

# Chargement du modèle XGBoost sauvegardé
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Best_XGBoost_Business_Model.pkl")
model = joblib.load(MODEL_PATH)

# Définition du seuil pour la classification
threshold = 0.222

# Définition du format des données d'entrée attendues
class Features(BaseModel):
    features: dict

# Endpoint de test pour vérifier que l'API fonctionne
@app.get("/test")
async def test():
    return {"message": "Bienvenue sur l'API FastAPI !"}

# Endpoint racine
@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API FastAPI ! - visitez /docs pour explorer"}

# Fonction pour calculer un score métier basé sur des coûts de faux positifs/faux négatifs
def calculate_business_score(y_proba, threshold=0.222, cost_fn=10, cost_fp=1):
    y_pred = int(y_proba >= threshold)
    y_true = [0] if y_pred == 0 else [1]
    y_proba_arr = [y_proba]
    y_pred_arr = [(p >= threshold) for p in y_proba_arr]
    fp = int((y_pred_arr[0] == 1) and (y_true[0] == 0))
    fn = int((y_pred_arr[0] == 0) and (y_true[0] == 1))
    total_cost = fp * cost_fp + fn * cost_fn
    worst_cost = cost_fn + cost_fp
    return 1 - (total_cost / worst_cost)

# Endpoint pour obtenir une prédiction basée sur les données envoyées
@app.post("/predict")
async def predict(data: Features, request: Request):
    X_input = np.array([list(data.features.values())])
    y_proba = model.predict_proba(X_input)[0][1]
    prediction = int(y_proba >= threshold)
    score_metier = calculate_business_score(y_proba)

    return {
        "proba_defaut": float(y_proba),
        "classe": "refusé" if prediction else "accepté",
        "score_metier": round(score_metier, 2)
    }
