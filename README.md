# Lab 1 – Étape 1 : Initialiser la structure du projet

Dans cette étape, nous créons la structure de base du projet pour préparer le pipeline MLOps.  
L'objectif est d'organiser les dossiers et fichiers nécessaires pour la suite du lab.

---

## Étapes réalisées

1. **Création du dossier de travail** :


mkdir mlops-lab-01
cd mlops-lab-01

Création des sous-dossiers et Création du fichier identifiant le modèle actif:

<img width="502" height="289" alt="img1" src="https://github.com/user-attachments/assets/feda445d-2661-498a-bb3f-5e5c1b7dd80f" />


 Arborescence attendue du projet:

<img width="334" height="176" alt="img2" src="https://github.com/user-attachments/assets/45c17569-3a0f-44f8-ae16-08de8dd9fb1d" />

# Lab 1 – Étape 2 : Préparer l'environnement Python

Dans cette étape, nous configurons l'environnement Python nécessaire pour exécuter tous les scripts du lab et installer les dépendances.

---

## Étapes réalisées

1. **Création de l'environnement virtuel** :


python -m venv venv_mlops
.\venv_mlops\Scripts\Activate.bat

<img width="446" height="38" alt="img6" src="https://github.com/user-attachments/assets/eb72b310-0dd9-4c59-833b-56709eb0b15a" />

Installation des dépendances nécessaires :

<img width="959" height="239" alt="img3" src="https://github.com/user-attachments/assets/ba4cb41b-a200-4032-8318-a9a2994267a1" />


# Lab 1 – Étape 3 : Génération du dataset

Dans cette étape, nous générons un dataset synthétique de churn client pour servir de base à l'entraînement d'un modèle.

## Implémentation du script `generate_data.py`

- Génération d’un dataset de **1200 clients** avec les colonnes suivantes :
  - `tenure_months` : ancienneté en mois
  - `num_complaints` : nombre de plaintes
  - `avg_session_minutes` : durée moyenne de session en minutes
  - `plan_type` : type d'abonnement (`basic` ou `premium`)
  - `region` : région (`NA`, `EU`, `AF`, `AS`)
  - `churn` : variable binaire indiquant si le client quitte le service (0 = reste, 1 = churn)

- La génération est **déterministe** grâce à une graine pseudo-aléatoire (`seed`) pour assurer la reproductibilité.

- La probabilité de churn est calculée via un **modèle logistique simple** basé sur les variables explicatives.

- Le script crée le dossier `data/` si nécessaire et sauvegarde le fichier CSV `raw.csv`.

## Exécution du script


python src/generate_data.py

<img width="587" height="150" alt="img12" src="https://github.com/user-attachments/assets/77842a69-b7ce-401f-b139-41fda99e3e82" />


# Lab 1 – Étape 4 : Préparation des données & contrôles qualité

Dans cette étape, nous prétraitons le dataset généré précédemment et effectuons des contrôles qualité avant l’entraînement du modèle.

## Implémentation du script `prepare_data.py`

Ce script réalise les opérations suivantes :

1. **Chargement du fichier brut** `data/raw.csv`.
2. **Nettoyage des données** :
   - Clip des valeurs négatives dans `avg_session_minutes` à 0 ;
   - Normalisation des champs catégoriels :
     - `plan_type` en minuscules et sans espaces ;
     - `region` en majuscules et sans espaces.
3. **Contrôles qualité** :
   - Vérification de la présence de toutes les colonnes attendues ;
   - Vérification que moins de 5 % des valeurs sont manquantes par colonne ;
   - Vérification du type numérique des colonnes quantitatives (`tenure_months`, `num_complaints`, `avg_session_minutes`).
4. **Sauvegarde** :
   - Dataset prétraité : `data/processed.csv` ;
   - Statistiques d’entraînement des colonnes numériques (`mean` et `std`) : `registry/train_stats.json`.

Ce module correspond à l’étape de **prétraitement** d’un pipeline MLOps avant l’entraînement d’un modèle.

## Exécution du script


python src/prepare_data.py


<img width="521" height="53" alt="img13" src="https://github.com/user-attachments/assets/251b0c24-cddc-4d26-9325-f7acab4da5ad" />

# Lab 1 – Étape 5 : Entraîner, versionner et valider le modèle

Dans cette étape, nous entraînons un modèle de churn, le versionnons et le validons via des métriques et un gate minimal.

## Implémentation du script `train.py`

Le script réalise les étapes suivantes :

1. **Chargement du dataset prétraité** `data/processed.csv`.
2. **Séparation des features et de la cible** (`churn`).
3. **Construction d’un pipeline scikit-learn** :
   - Prétraitement :
     - `StandardScaler` pour les colonnes numériques ;
     - `OneHotEncoder` pour les colonnes catégorielles ;
   - Modèle : régression logistique binaire.
4. **Split train / test stratifié**.
5. **Entraînement du modèle** sur les données d’entraînement.
6. **Évaluation des métriques** : accuracy, precision, recall, F1-score.
7. **Comparaison avec une baseline** (prédiction toujours 0).
8. **Sauvegarde** :
   - Modèle entraîné dans `models/` ;
   - Métadonnées (métriques, seed, version, etc.) dans `registry/metadata.json` ;
   - Mise à jour du fichier `registry/current_model.txt` si le modèle passe le gate F1 minimal.

Ce module illustre un pipeline MLOps minimaliste pour le training et le versioning de modèles.

## Exécution du script


python src/train.py


<img width="523" height="124" alt="img14" src="https://github.com/user-attachments/assets/f12a41de-e877-471c-8298-7422514f56a5" />

# Lab 1 – Étape 6 : Inspecter la registry et le modèle courant

Dans cette étape, nous entraînons à nouveau le modèle avec un composant de **tuning du seuil optimal** (F1), puis nous inspectons la registry et mettons à jour le modèle courant.

## Implémentation du script `evaluate.py`

Le script réalise les étapes suivantes :

1. **Chargement du dataset prétraité** `data/processed.csv`.
2. **Construction du pipeline scikit-learn** :
   - Standardisation des variables numériques ;
   - OneHotEncoding des variables catégorielles ;
   - Régression logistique.
3. **Split train/test stratifié**.
4. **Entraînement du modèle**.
5. **Évaluation des métriques** :
   - Accuracy, precision, recall, F1 avec seuil par défaut = 0.5 ;
   - Recherche du **seuil optimal** maximisant la F1 ;
   - Calcul d’une **baseline triviale** (prédiction toujours 0).
6. **Sauvegarde** :
   - Modèle entraîné dans `models/` ;
   - Métadonnées dans `registry/metadata.json` ;
   - Mise à jour du modèle courant dans `registry/current_model.txt` si le gate F1 est validé.

Ce script illustre une étape “Train + Eval + Register” avec **optimisation simple du seuil**, comme dans un vrai pipeline MLOps.

## Exécution du script


python src/evaluate.py


<img width="710" height="189" alt="img5" src="https://github.com/user-attachments/assets/62e5f088-34ba-494e-9574-e9ff8b210137" />

# Lab 1 – Étape 7 : Créer une API `/predict` avec le modèle courant

Cette étape met en place une **API FastAPI** pour servir le modèle de churn versionné.

## Fonctionnalités du service

- Charge dynamiquement le **modèle courant** indiqué dans `registry/current_model.txt`.
- Endpoint **`/health`** : vérifie l'état de l'API et la présence du modèle courant.
- Endpoint **`/predict`** : prédit le churn à partir de features d’un client :
  - `tenure_months`, `num_complaints`, `avg_session_minutes`
  - `plan_type` (basic, premium…)
  - `region` (NA, EU, AF, AS…)
- **Journalisation** de chaque requête de prédiction dans `logs/predictions.log` au format JSON (une ligne par prédiction).

Cette API illustre une étape **Serve** dans un pipeline MLOps minimal : un modèle versionné est promu dans la registry, puis utilisé par un service d’inférence léger.

## Création du fichier


New-Item src/api.py

<img width="652" height="301" alt="img7" src="https://github.com/user-attachments/assets/35593400-f38b-4ed1-99ec-a7e06368d819" />
<img width="908" height="467" alt="img8" src="https://github.com/user-attachments/assets/2376bdea-d5ec-450f-bf0f-5f3ed3b5fad2" />

# Lab 1 – Étape 8 : Détection de drift des données via les logs

Cette étape permet de détecter un **data drift** simple sur les features d’entrée à partir des logs de prédiction.

---

## Fonctionnalités du script `monitor_drift.py`

- Charge les **statistiques d’entraînement** (moyenne / écart-type) depuis `registry/train_stats.json`.
- Charge les **logs de prédiction récents** (`logs/predictions.log`), une prédiction par ligne JSON.
- Compare la moyenne des features observées en production aux moyennes d’entraînement via un **score Z** :

<img width="554" height="136" alt="img9" src="https://github.com/user-attachments/assets/311d1373-37b6-4096-a697-9824108fdb64" />



# Lab 1 – Étape 9 : Gestion des versions du modèle et rollback
Cette étape illustre comment gérer plusieurs versions d’un modèle dans un registry minimaliste et revenir à une version précédente si nécessaire.
Exemple : entraîner la version v2 avec gate F1 à 0.70 

Créer le script de rollback
New-Item src/rollback.py

<img width="875" height="202" alt="img10" src="https://github.com/user-attachments/assets/56323137-9972-4ffc-9d9a-3de972c3bb20" />
<img width="638" height="83" alt="img11" src="https://github.com/user-attachments/assets/99361757-38b4-4c5b-b38b-c62d921fb5e5" />
