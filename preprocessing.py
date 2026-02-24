import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from fonctions import CustomTargetEncoder

def preprocess_data():

    print("1. Chargement des données pré-nettoyées...")
    df_train = pd.read_csv('Train_cleaned.csv')
    df_test_final = pd.read_csv('Test_cleaned.csv')

    X = df_train.drop('Price', axis=1)
    y = df_train['Price']

    print("2. Split Train/Validation (80% Train, 20% Validation)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

    categorical_columns = ['Airline', 'Source', 'Destination']

    print(f"3. Encodage avec notre Algorithme Maison (Lissage = 20) sur {categorical_columns}...")
    encoder = CustomTargetEncoder(cols=categorical_columns, smoothing=20)
    
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_val_encoded = encoder.transform(X_val)
    X_test_final_encoded = encoder.transform(df_test_final)

    print("4. Standardisation des variables pour préparer le terrain au Deep Learning...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    
    X_val_scaled = scaler.transform(X_val_encoded)
    X_test_final_scaled = scaler.transform(X_test_final_encoded)

    print("5. Sauvegarde des matrices Numpy pour l'entraînement Machine Learning et Deep Learning...")
    joblib.dump((X_train_scaled, X_val_scaled, y_train, y_val), 'ready_data.pkl')
    joblib.dump(X_test_final_scaled, 'ready_test_data.pkl')
    print("Prétraitement terminé ! Données sérialisées dans 'ready_data.pkl'.")

