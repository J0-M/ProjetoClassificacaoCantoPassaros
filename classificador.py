import os
import numpy as npy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report

def main():
    dataframePath = "dataframeSegmentado.pkl"

    if os.path.exists(dataframePath):
        with open("dataframeSegmentado.pkl", "rb") as readFile:
            df = pickle.load(readFile)
            print("Dataframe carregado com sucesso!")

    X = df.drop(columns=["roi_label"]) # x = features
    y = df["roi_label"] # y = passaros

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100) #80% treino, 20% teste

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # normaliza os dados das features para que o treino não seja enviesado por valores distoantes
    X_test_scaled = scaler.transform(X_test)

    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    y_pred = knn.predict(X_test_scaled) #testa o knn com o conjunto 20% teste

    f1 = f1_score(y_test, y_pred, average="weighted") #f1 score

    print(f"K: {k}")
    print(f"F1-score do KNN: {f1:.2f}")
    print(classification_report(y_test, y_pred))

    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn, f) #salva knn num pickle

    print("Modelo salvo como knn_model.pkl")
    
if __name__ == '__main__':
    main()
