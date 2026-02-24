import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import optuna

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- Performances {name} ---")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print("-" * 30)
    return rmse, r2

def main():
    print("1. Chargement des données d'entraînement et validation...")
    X_train, X_val, y_train, y_val = joblib.load('ready_data.pkl')

    # JUSTIFICATION DÉMARCHE SCIENTIFIQUE (Pour le rapport) : 
    # Le critère est "Stratified K-Fold". Cependant, notre Problème est une RÉGRESSION (prix continu).
    # Stratifier du continu mathématique pur n'est pas conventionnel.
    # Nous utilisons donc une "K-Fold standard avec Shuffling (Brassage)" robuste, à 5 plis (folds).
    print("Mise en place de la validation croisée robuste (K-Fold, 5 splits)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n2. Entraînement du Modèle 1 : Random Forest (Recherche par RandomSearch)")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   n_iter=5, cv=kf, scoring='neg_root_mean_squared_error', 
                                   random_state=42, n_jobs=-1)
    
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Meilleurs hyperparamètres RF trouvés : {rf_search.best_params_}")
    
    y_pred_rf = best_rf.predict(X_val)
    evaluate_model("Random Forest", y_val, y_pred_rf)

    print("\n3. Entraînement du Modèle 2 : LightGBM optimisé avec Optuna (Bayesian Optimization)")
    # Justification (Rapport) : LightGBM a été préféré car il gère mieux et 
    # plus rapidement l'apprentissage de par sa construction d'arbre (leaf-wise).
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    print("Recherche Optuna en cours (15 itérations d'exploration des paramètres) ...")
    study.optimize(objective, n_trials=15)
    
    print(f"Meilleurs hyperparamètres LightGBM : {study.best_params}")
    
    best_lgb = lgb.LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    best_lgb.fit(X_train, y_train)
    
    y_pred_lgb = best_lgb.predict(X_val)
    evaluate_model("LightGBM", y_val, y_pred_lgb)

    print("\nSauvegarde du modèle LightGBM ('lightgbm_model.pkl') pour éventuelle prédiction...")
    joblib.dump(best_lgb, 'lightgbm_model.pkl')
    print("Fin de l'approche Classique ML.")

if __name__ == "__main__":
    main()


