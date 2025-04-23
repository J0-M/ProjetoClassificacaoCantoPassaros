import os
import numpy as npy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, get_scorer_names, get_scorer

def melhorK(ks, X_treino, X_val, y_treino, y_val, X_teste, y_teste):
    acuracias_val = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        acuracias_val.append(f1_score(y_val, pred, average="macro"))
        
    melhor_val = max(acuracias_val)
    melhor_k = ks[npy.argmax(acuracias_val)]        
    knn = KNeighborsClassifier(n_neighbors=melhor_k)
    knn.fit(npy.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    print("Melhor k na validação: %d (acc=%.2f)" % (melhor_k, melhor_val))
    pred = knn.predict(X_teste)
    print("acurácia no teste: %.2f" % (f1_score(y_teste, pred, average="macro")))
    
    #with open("melhorK_knn.pkl", "wb") as file:
        #pickle.dump(melhor_k, file)
    
    return knn, melhor_k, melhor_val

def calculaK(X, y):
    melhorKPath = "melhorK_knn.pkl"
    
    if os.path.exists(melhorKPath):
        with open("melhorK_knn.pkl", "rb") as readFile:
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
        
        _, k, _ = melhorK(ks, X_treino, X_val, y_treino, y_val, X_teste, y_teste)
        print(f"k = {k}")
    
    return k

def treinaKNN(k, X_train_scaled, y_train):
    knnPath = "knnModel.pkl"
    
    if os.path.exists(knnPath):
        with open("knnModel.pkl", "rb") as readFile:
            knn = pickle.load(readFile)
            print("KNN carregado com sucesso!")
    else:
        knn = KNeighborsClassifier(n_neighbors=k)
        with open("knnModel.pkl", "wb") as f:
            pickle.dump(knn, f) #salva knn num pickle

        print("Modelo salvo como knn_model.pkl")

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    return knn

def printResultados(knn, X_test_scaled, y_test):
    y_pred = knn.predict(X_test_scaled) #testa o knn com o conjunto 20% teste

    f1 = f1_score(y_test, y_pred, average="weighted") #f1 score
        
    print(f"F1-score do KNN: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    
def knnCruzado(X, y):
    #a validação cruzada será realizada em 10 vias.
    k_vias = 10
    
    #filtra os y, e elimina as amostras que tem menos que 2 especies
    counts = y.value_counts()
    classes_validas = counts[counts >= k_vias].index
    filtro = y.isin(classes_validas)
    X = X[filtro]
    y = y[filtro]

    #usar o protocolo de validação cruzada estratificada
    skf = StratifiedKFold(n_splits=k_vias, shuffle=True, random_state=10)

    acuracias = []

    #a função split retorna os índices das instâncias que devem ser usadas para o treinamento e o teste.
    for idx_treino, idx_teste in skf.split(X, y):
        
        #extrair as instâncias de treinamento de acordo com os índices fornecidos pelo skf.split
        X_treino = X.iloc[idx_treino]
        y_treino = y.iloc[idx_treino]
        
        #extrair as instâncias de teste de acordo com os índices fornecidos pelo skf.split
        X_teste = X.iloc[idx_teste]
        y_teste = y.iloc[idx_teste]
        
        #separar as instâncias de treinamento entre treinamento e validação para a otimização do hiperparâmetro k
        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, test_size=0.2, stratify=y_treino, shuffle=True, random_state=10)
        
        #colocar todas as variáveis na mesma escala, usando o conjunto de treinamento para calcular os parâmetros da escala
        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        #escolher o k com o melhor resultado no conjunto de validação e treinar o KNN com o melhor k.
        knn, _, _ = melhorK(range(1,30,2), X_treino, X_val, y_treino, y_val, X_teste, y_teste)
        
        y_pred = knn.predict(X_teste)
        
        #calcular a acurácia no conjunto de testes desta iteração e salvar na lista.
        
        acuracias.append(f1_score(y_teste, y_pred, average="macro"))
        
    
    return knn, acuracias
    

def main():
    dataframePath = "dataframeSegmentado.pkl"

    if os.path.exists(dataframePath):
        with open("dataframeSegmentado.pkl", "rb") as readFile:
            df = pickle.load(readFile)
            print("Dataframe carregado com sucesso!")

    X = df.drop(columns=["roi_label"]) # x = features
    y = df["roi_label"] # y = passaros

    print(X.shape)
    
    #####################################

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2000) #80% treino, 20% teste

    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train) # normaliza os dados das features para que o treino não seja enviesado por valores distoantes
    #X_test_scaled = scaler.transform(X_test)
    
    #####################################
    
    #k = calculaK(X, y)
    #knn = treinaKNN(k, X_train_scaled, y_train)

    #printResultados(knn, X_test_scaled, y_test)
    
    #####################################
    
    knnCruzadoPath = "knnCruzado.pkl"
    acuraciasPath = "acuracias.pkl"
    
    if os.path.exists(knnCruzadoPath) and os.path.exists(acuraciasPath):
        with open("knnCruzado.pkl", "rb") as readFile:
            knn = pickle.load(readFile)
            print("KNN com validação cruzada carregado com sucesso!")
            
        with open("acuracias", "rb") as readFile:
            acuracias = pickle.load(readFile)
            print("Acuracias carregados com sucesso!")
    else:
        knn, acuracias = knnCruzado(X, y)
    
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f" % (min(acuracias), max(acuracias), npy.mean(acuracias), npy.std(acuracias)))

    
if __name__ == '__main__':
    main()
