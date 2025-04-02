import os
import numpy as npy
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report

def melhorK(ks, X_treino, X_val, y_treino, y_val, X_teste, y_teste):
    acuracias_val = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        acuracias_val.append(accuracy_score(y_val, pred))
        
    melhor_val = max(acuracias_val)
    melhor_k = ks[npy.argmax(acuracias_val)]        
    knn = KNeighborsClassifier(n_neighbors=melhor_k)
    knn.fit(npy.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    print("Melhor k na validação: %d (acc=%.2f)" % (melhor_k, melhor_val))
    pred = knn.predict(X_teste)
    print("acurácia no teste: %.2f" % (accuracy_score(y_teste, pred)))
    
    with open("melhorK.pkl", "wb") as file:
        pickle.dump(melhor_k, file)
    
    return melhor_k

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
    
    melhorKPath = "melhorK.pkl"
    
    if os.path.exists(melhorKPath):
        with open("melhorK.pkl", "rb") as readFile:
            k = pickle.load(readFile)
            print(f"K carregado, k = {k}")
    else:
        print("Calculando melhor K...")
        ks = list(range(1,30,2))
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=3000)
        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, test_size=0.2, random_state=3000)
        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)
        
        k = melhorK(ks, X_treino, X_val, y_treino, y_val, X_teste, y_teste)
        print(f"k = {k}")
    
    #####################################

    knn = KNeighborsClassifier(n_neighbors=k) # k = 5, pelos testes

    knn.fit(X_train_scaled, y_train) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    y_pred = knn.predict(X_test_scaled) #testa o knn com o conjunto 20% teste

    f1 = f1_score(y_test, y_pred, average="weighted") #f1 score
    
    print(f"F1-score do KNN: {f1:.2f}")
    print(classification_report(y_test, y_pred))

    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn, f) #salva knn num pickle

    print("Modelo salvo como knn_model.pkl")
    
if __name__ == '__main__':
    main()
