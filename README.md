# Rapport de Projet : Prédiction des Prix de Billets d'Avion (Régression)

## 1. Objectif et Démarche Scientifique

Ce projet a pour but de résoudre un problème de **Régression** : prédire le prix d'un billet d'avion en fonction de ses caractéristiques (Compagnie, Dates, Escalier, Trajet).
Pour répondre aux exigences de qualité professionnelle, le projet est découpé en scripts modulaires (`.py`) et respecte scrupuleusement le principe de séparation des données. Une approche de **"Data Leakage Prevention"** stricte a été appliquée : le jeu de `Test` n'est jamais analysé ou utilisé lors du `fit` des encodeurs/scalers.

## 2. Exploration des Données (EDA) - `data-analyse.py`

### Analyse Critique et Piège du Dataset

L'exploration experte (EDA) a mis en évidence un **bruit critique dans les données brutes** (`Data_Train.csv`). À partir d'une certaine ligne, le séparateur du fichier CSV passe subitement d'une virgule (`,`) à un point-virgule (`;`).  
**Action :** Une lecture basique avec Pandas détruisait les données (générant des milliers de `NaN`). J'ai implémenté une expression régulière `sep=',|;'` pour lire le fichier de manière robuste, ce qui a sauvé 100% de la base de données. Il ne restait alors plus qu'une ou deux lignes corrompues qui ont été supprimées.

### Visualisations

- **Distribution des Prix (`distribution_prix.png`) :** La distribution est asymétrique avec un étalement vers la droite (Skewed to the right). La majorité des prix se concentrent autour de 5000-10000 Roupies, avec des outliers très marqués pouvant dépasser les 50000.
- **Boxplot par Compagnie (`prix_par_compagnie.png`) :** Permet de constater de graves disparités. Par exemple, "Jet Airways Business" a des prix statistiquement très distincts des compagnies low-cost (SpiceJet, IndiGo), une information cruciale pour le modèle.

## 3. Prétraitement et Feature Engineering - `data-cleaning.py`

Les modèles d'apprentissage automatique ne traitant pas le texte brut, une phase d'Ingénierie des Caractéristiques a été réalisée :

- **Variables temporelles :** Extraction du mois, du jour, de l'heure et des minutes pour les champs de départ et d'arrivée. On note l'exclusion délibérée de l'Année (car la variance est nulle, tous les vols sont en 2019).
- **Durée des vols :** Conversion du texte de type "2h 50m" en deux caractéristiques numériques propres `Duration_Hours` et `Duration_Mins`.
- **Encodage Ordinal :** Transformation mathématique logique de la variable `Total_Stops` (ex: "non-stop" -> 0, "1 stop" -> 1).
- **Réduction du bruit :** Suppression de colonnes génératrices de multicolinéarité (`Route` corrélée à `Total_Stops`) et de la colonne `Additional_Info` qui contient majoritairement l'assertion "No info".

## 4. Composante Personnalisée et Sérialisation - `fonctions.py` & `preprocessing.py`

C'est ici qu'intervient l'implémentation algorithmique "**Custom**" majeure de ce projet.

### Implémentation du _CustomTargetEncoder_ avec Lissage Bayésien

J'ai conçu une classe `CustomTargetEncoder` pour remplacer les outils standard de Scikit-Learn face à un problème précis de notre base :

- **Le problème :** Un _One-Hot Encoding_ aurait fait exploser le nombre de colonnes (Fléau de la dimensionnalité) à cause du nombre important de routes. De son côté, un _Target Encoding_ standard (qui substitue le nom de la compagnie par la moyenne de ses prix) provoque un sur-apprentissage grave sur les compagnies très rares ("Trujet" ne contenant qu'un paramètre).
- **La solution :** Un Target Encoder pondéré (Smoothing). Plus la compagnie apparaît rarement dans le dataset, plus son encodage sera repoussé et lissé vers la **Moyenne globale des prix**. Cela protège mathématiquement nos algorithmes des aberrations.

### Flux de Validation Scientifique

1. **Splitting :** Isolement de 20% des données en Set de Validation (80:20). L'encodage "Custom" et la standardisation (`StandardScaler`) sont "fittés" uniquement sur les 80% (Train) pour proscrire toute fuite mathématique (_Data Leakage_).
2. **Scaling :** Standardisation impérative des données en préparation de l'approche Deep Learning (qui nécessite des entrées centrées-réduites sous peine d'explosion du gradient d'erreur).

## 5. Machine Learning Classique - `approche-ML.py`

J'ai choisi de confronter deux des algorithmes les plus puissants du marché sur ce type de données tabulaires :

- **Modèle 1 : RandomForestRegressor**
  - **Recherche :** RandomSearch (pour limiter le coût calculatoire).
  - **Résultat :** R² ≈ 81.44 % | RMSE ≈ 2000
- **Modèle 2 : LightGBMRegressor**
  - **Recherche :** Optimisation bayésienne par **Optuna**. Optuna est algorithmiquement supérieur au GridSearch classique car il intègre les gradients d'erreurs passés pour converger vers la zone idéale des hyperparamètres.
  - **Pourquoi LightGBM plutôt que XGBoost ?** XGBoost souffre de fortes contraintes de dépendances C++ (OpenMP pour Mac) et LightGBM compile ses arbres horizontalement (leaf-wise), le rendant bien plus preste pour un résultat supérieur.
  - **Résultat :** R² ≈ 82.07 % | RMSE ≈ 1966

_NB sur le Validation-Set (KFold) : Bien que le cahier des charges propose un Stratified K-Fold, il convient d'analyser critiquement que nous sommes face à une Régression (continu). Une méthode en K-Fold simple statique (shuffle=True) est la norme de validation scientifique reconnue ici._

## 6. Approche Deep Learning - `approche-DL.py`

Développement sous `TensorFlow/Keras` d'un Réseau de Neurones (MLP).

- **Architecture :** Denses imbriquées (128 -> 64 -> 32) afin d'apprendre les dimensions cachées les plus intriquées des variables de l'aviation.
- **Régulation :** Insertion de couches _Dropout_ à 20% désactivant aléatoirement des synapses pour inhiber le mémoriel (Overfitting). La couche sortante étant une valeur prédictive linéaire (activation=linear).
- **Early Stopping :** Surveillance de la courbe de Validation (val_loss) pendant le _fit_, enjoignant à l'algorithme de figer ses poids si après 10 époques (patience=10) aucune régression relative de l'erreur n'est observée. Les poids d'Or (Best Weights) sont alors restaurés.
- **Résultat :** R² ≈ 74.11 % | RMSE ≈ 2362
  _(Une capture de la matrice d'apprentissage a été sauvegardée dans `learning_curve_dl.png` visualisant l'évolution conjointe des Loss (Entraînement/Validation) dans le temps)._

## 7. Analyse et Conclusion

L'expérience confirme un postulat classique du domaine de la Data Science : sur les datasets structurés tabulaires d'une volumétrie moyenne (< 1 M lignes), les algorithmes d'ensemble par Gradient Boosting tendent à vaincre structurellement le Deep Learning pur.
En témoigne notre Gradient Booster (`LightGBM`), armé de sa calibration via l'optimiseur Bayésien Optuna, qui achève la meilleure généralisation avec **82%** de variance expliquée et la plus faible erreur monétaire (RMSE). Ce modèle a été exporté en sérialisation `.pkl` et peut d'ores et déjà être déployé en architecture de production.
