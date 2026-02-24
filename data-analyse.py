import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

train_path = 'Data_Train.csv'
df_train = pd.read_csv(train_path, sep=',|;', engine='python')

print("="*50)
print("Aperçu des 5 premières lignes :")
print(df_train.head())

print("\n" + "="*50)
print("Informations générales sur le dataset :")
print(df_train.info())

print("\n" + "="*50)
print("Nombre de valeurs manquantes par colonne :")
nan_counts = df_train.isnull().sum()
print(nan_counts[nan_counts > 0])

print("\n" + "="*50)
print("Statistiques descriptives du Prix :")
print(df_train['Price'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(df_train['Price'], bins=50, kde=True, color='blue')
plt.title('Distribution des prix des billets d\'avion')
plt.xlabel('Prix')
plt.ylabel('Fréquence')
plt.savefig('distribution_prix.png')
print("Graphique de distribution sauvegardé sous 'distribution_prix.png'.")

plt.figure(figsize=(12, 6))
sns.boxplot(y='Airline', x='Price', data=df_train.sort_values('Price', ascending=False))
plt.title('Prix en fonction de la compagnie aérienne')
plt.tight_layout()
plt.savefig('prix_par_compagnie.png')
print("Graphique Boxplot sauvegardé sous 'prix_par_compagnie.png'.")

