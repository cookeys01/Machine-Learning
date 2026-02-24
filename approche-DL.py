import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def main():
    print("1. Chargement des matrices prêtes pour le Deep Learning...")
    X_train, X_val, y_train, y_val = joblib.load('ready_data.pkl')
    

    print("\n2. Construction de l'architecture du Réseau de Neurones...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # JUSTIFICATION EARLY STOPPING : Si l'erreur de validation stagne pendant 10 epochs,
    # on arrête l'entrainement et on restaure les meilleurs poids (Contre l'overfitting).
    print("\n3. Entraînement avec surveillance (EarlyStopping) pour éviter l'Overfitting...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n4. Évaluation des performances du Deep Learning...")
    y_pred_dl = model.predict(X_val)
    y_pred_dl = y_pred_dl.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_dl))
    r2 = r2_score(y_val, y_pred_dl)
    print(f"--- Performances Réseau de Neurones ---")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    
    print("\n5. Sauvegarde de la courbe d'apprentissage ('learning_curve_dl.png')...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Erreur Train (Loss)')
    plt.plot(history.history['val_loss'], label='Erreur Validation (Val Loss)')
    plt.title("Analyse de l'Apprentissage du Réseau de Neurones")
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (Prix^2)')
    plt.legend()
    plt.savefig('learning_curve_dl.png')
    
    print("Modèle Deep Learning ('dl_model.h5') sauvegardé !")
    model.save('dl_model.h5')

if __name__ == "__main__":
    main()

