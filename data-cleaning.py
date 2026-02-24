import pandas as pd
import numpy as np

# On Nettoie le dataset et extrait des features exploitables par un modèle ML.


def clean_and_engineer_features(df):
   
    df = df.copy()
    
    df.dropna(inplace=True)
    
    df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
    df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month
    df.drop('Date_of_Journey', axis=1, inplace=True)
    
    df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
    df['Dep_Min'] = pd.to_datetime(df['Dep_Time']).dt.minute
    df.drop('Dep_Time', axis=1, inplace=True)
    
    df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
    df['Arrival_Min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
    df.drop('Arrival_Time', axis=1, inplace=True)
    
    duration_list = list(df['Duration'])
    for i in range(len(duration_list)):
        if len(duration_list[i].split()) != 2:
            if "h" in duration_list[i]:
                duration_list[i] = duration_list[i].strip() + " 0m"
            else:
                duration_list[i] = "0h " + duration_list[i]
                
    duration_hours = []
    duration_mins = []
    for i in range(len(duration_list)):
        duration_hours.append(int(duration_list[i].split(sep = "h")[0]))
        duration_mins.append(int(duration_list[i].split(sep = "m")[0].split()[-1]))
        
    df['Duration_Hours'] = duration_hours
    df['Duration_Mins'] = duration_mins
    df.drop('Duration', axis=1, inplace=True)
    
    stops_mapping = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    df['Total_Stops'] = df['Total_Stops'].map(stops_mapping)
    
    df.drop('Route', axis=1, inplace=True)
    df.drop('Additional_Info', axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    print("Chargement des données brutes...")
    df_train = pd.read_csv('Data_Train.csv', sep=',|;', engine='python')
    df_test = pd.read_csv('Test_set.csv', sep=',|;', engine='python')
    
    print("Début du nettoyage et feature engineering...")
    df_train_clean = clean_and_engineer_features(df_train)
    df_test_clean = clean_and_engineer_features(df_test)
    
    print("Sauvegarde des données nettoyées...")
    df_train_clean.to_csv('Train_cleaned.csv', index=False)
    df_test_clean.to_csv('Test_cleaned.csv', index=False)
    
    print("Nettoyage terminé avec succès ! Fichiers générés.")

