## Credit Scoring Engine – Implémentation MLOps complète

##
Contexte:

Ce projet a été réalisé dans le cadre d’une mission au sein de l’entreprise fictive "Prêt à dépenser", spécialisée dans les crédits à la consommation pour des clients sans historique de crédit.
L’objectif est de développer un outil de scoring crédit, permettant de prédire la probabilité de remboursement d’un client et de classifier les demandes en crédit accordé ou refusé, tout en suivant une démarche MLOps complète.

## 🎯 Objectifs du projet
- Élaborer un modèle de classification supervisée robuste et interprétable.

- Créer un score métier basé sur le coût différencié des erreurs (faux négatifs / faux positifs).

- Mettre en œuvre une infrastructure MLOps avec versionning, CI/CD et monitoring.

- Déployer le modèle via une API FastAPI connectée à une interface utilisateur Streamlit.

- Suivre la dérive des données (data drift) à l’aide de la librairie Evidently.

## Stack technique
- Python (Pandas, Scikit-Learn, XGBoost)

- FastAPI (API Backend)

- Streamlit (Frontend de test)

- MLFlow (Tracking & Model Registry)

- GitHub Actions (CI/CD)

- Render (Hébergement cloud API + UI)

- Evidently (Monitoring des données)

- Pytest (Tests unitaires)

## Pipeline MLOps

- 1. Entraînement & suivi des modèles
Tracking des expériences avec MLFlow.

- Stockage centralisé des modèles dans le Model Registry.

- Optimisation via GridSearchCV.

- Évaluation avec AUC, F1-score et un score métier basé sur les coûts d’erreur.

## 2. Feature Engineering
- Création de variables métiers (e.g. Credit/Annuity, EXT_SOURCE).

- Gestion des déséquilibres de classes avec SMOTE.

- Construction de caractéristiques polynomiales et métiers.

## 3. Sélection du meilleur modèle
- Comparaison de régression logistique, forêt aléatoire et XGBoost.

- Meilleur modèle retenu : XGBoost avec features métiers.

- Seuil de décision optimisé pour maximiser le score métier.

## 4. Déploiement
- API créée avec FastAPI, déployée sur Render.

- Interface Streamlit connectée à l’API.

- CI/CD avec GitHub Actions.

## 5. Monitoring
- Simulation du déploiement avec application_test comme données de production.

- Suivi du data drift avec Evidently sur 20 features.

##
Score métier
- FP (faux positifs) = clients fiables refusés → coût = 1

- FN (faux négatifs) = clients à risque acceptés → coût = 10

- Le score métier est défini par :
score = 1 - (coût_total / coût_max)

- Optimisation du seuil de classification pour réduire ce coût global.

## Résultats clés
- Meilleur AUC : 0.756

- Meilleur score métier : 0.641

- Bon équilibre entre performance globale et impact métier

- Modèle robuste mais amélioration possible du rappel


## Lien de démonstration
- API (FastAPI) 
- Interface (Streamlit)

## Test unitaire
- Tests réalisés :

- Vérification que le modèle utilise bien les 20 features sélectionnées

- Vérification de l’exécution sur un batch de données

- Vérification que les probabilités sont bien entre 0 et 1

## Explicabilité
- Importance globale des variables via SHAP

- Explications locales pour chaque prédiction (exemples individuels)

## Auteur
- Oumou Faye
- Projet réalisé dans le cadre du parcours Data Scientist CentraleSupélec - OpenClassrooms
- Mentor : Medina Hadjem
