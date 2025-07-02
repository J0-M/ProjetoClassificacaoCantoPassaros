import os
import numpy as npy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score

from datetime import datetime

def melhorK(ks, X_treino, X_val, y_treino, y_val, X_teste, y_teste, ka):
    acuracias_val = []
    topkKScores = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        proba = knn.predict_proba(X_val)
        
        acuracias_val.append(f1_score(y_val, pred, average="macro"))
        #topkKScores.append(top_k_accuracy_score(y_val, proba, k =ka))
        
    melhor_val = max(acuracias_val)
    #melhor_val = max(topkKScores)
    melhor_k = ks[npy.argmax(acuracias_val)]
    #melhor_k = ks[npy.argmax(topkKScores)] 

    knn = KNeighborsClassifier(n_neighbors=melhor_k)
    knn.fit(npy.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    print("Melhor k na validação: %d (acc=%.2f)" % (melhor_k, melhor_val))
    pred = knn.predict(X_teste)
    print("Acurácia no teste: %.2f" % (f1_score(y_teste, pred, average="macro")))
    #print("Acurácia no teste: %.2f" % (top_k_accuracy_score(y_teste, proba, k=ka)))
    
    return knn, melhor_k, melhor_val

def calculaK(X, y):
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
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    return knn

def printResultados(knn, X_test_scaled, y_test, ka):
    y_proba = knn.predict_proba(X_test_scaled)
    y_pred = knn.predict(X_test_scaled)

    f1 = f1_score(y_test, y_pred, average="macro")
    topk_acc = top_k_accuracy_score(y_test, y_proba, k=ka, labels=knn.classes_)

    print(f"F1-score do KNN: {f1:.2f}")
    print(f"Top-{ka} Accuracy: {topk_acc:.2f}")
    #print(classification_report(y_test, y_pred))
    
def knnCruzado(X, y, ka):
    #a validação cruzada será realizada em 10 vias.
    k_vias = 10
    
    #usar o protocolo de validação cruzada estratificada
    skf = StratifiedKFold(n_splits=k_vias, shuffle=True, random_state=10)
    
    acuracias = []
    topKScores = []
    
    #filtra os y, e elimina as amostras que tem menos que k_vias especies
    counts = y.value_counts()
    classes_validas = counts[counts >= k_vias].index
    filtro = y.isin(classes_validas)
    X = X[filtro]
    y = y[filtro]
    
    #print("Quantidade de amostras: ", X.shape)
    #num_especies = y.nunique()
    #print("Quantidade de especies: ", num_especies)
    
    matrizFoldPath = "matrizesProba_knn_treinoSegmentado"
    if(not os.path.exists(matrizFoldPath)):
        os.makedirs(matrizFoldPath, exist_ok=True)
        
    modelosFoldPath = "modelos_knn_treinoSegmentado"
    if(not os.path.exists(modelosFoldPath)):
        os.makedirs(modelosFoldPath, exist_ok=True)

    #a função split retorna os índices das instâncias que devem ser usadas para o treinamento e o teste.
    for foldId, (idx_treino, idx_teste) in enumerate(skf.split(X, y)):
        
        matriz_filename = os.path.join(matrizFoldPath, f"matriz_{foldId + 1}.pkl")
        modelo_filename = os.path.join(modelosFoldPath, f"knn_model_fold_{foldId + 1}.pkl")
        
        #extrair as instâncias de treinamento de acordo com os índices fornecidos pelo skf.split
        X_treino = X.iloc[idx_treino]
        y_treino = y.iloc[idx_treino]
        
        #conjunto treino foldId
        
        #extrair as instâncias de teste de acordo com os índices fornecidos pelo skf.split
        X_teste = X.iloc[idx_teste]
        y_teste = y.iloc[idx_teste]
        
        #conjunto teste foldId
        
        path_folds = "folds_audiosSegmentados_knn"
        fold_archive_X_treino = os.path.join(path_folds, f"X_treino_fold_{foldId + 1}.pkl")
        fold_archive_y_treino = os.path.join(path_folds, f"y_treino_fold_{foldId + 1}.pkl")
        fold_archive_X_teste = os.path.join(path_folds, f"X_teste_fold_{foldId + 1}.pkl")
        fold_archive_y_teste = os.path.join(path_folds, f"y_teste_fold_{foldId + 1}.pkl")
        
        if(not os.path.exists(path_folds)):
            os.makedirs(path_folds, exist_ok=True)
            
        with open(fold_archive_X_treino, "wb") as f:
            pickle.dump(X_treino, f)
        print(f"Fold salvos em {fold_archive_X_treino}")
        
        with open(fold_archive_y_treino, "wb") as f:
            pickle.dump(y_treino, f)
        print(f"Fold salvos em {fold_archive_y_treino}")
        
        with open(fold_archive_X_teste, "wb") as f:
            pickle.dump(X_teste, f)
        print(f"Fold salvos em {fold_archive_X_teste}")
        
        with open(fold_archive_y_teste, "wb") as f:
            pickle.dump(y_teste, f)
        print(f"Fold salvos em {fold_archive_y_teste}")
        
        if(os.path.exists(modelo_filename)):
            print(f"Carregando modelo do fold {foldId + 1}...")
            with open(modelo_filename, "rb") as f_modelo:
                knn = pickle.load(f_modelo)
            
            ss = StandardScaler()
            ss.fit(X_treino)
            X_teste = ss.transform(X_teste)
            y_pred = knn.predict(X_teste)
            y_proba = knn.predict_proba(X_teste)
            
            if os.path.exists(matriz_filename):
                print(f"Carregando matriz do fold {foldId + 1}...")
                with open(matriz_filename, "rb") as f:
                    matriz_info = pickle.load(f)
            else:
                matriz_info = {
                    "fold": foldId,
                    "y_true": y_teste.values,
                    "y_proba": y_proba,
                    "classes": knn.classes_
                }
                with open(matriz_filename, "wb") as f:
                    pickle.dump(matriz_info, f)
                print(f"Probabilidades salvas em {matriz_filename}")
        else:
            print(f"Criando modelo do fold {foldId + 1}...")
            
            #separar as instâncias de treinamento entre treinamento e validação para a otimização do hiperparâmetro k
            X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, test_size=0.2, stratify=y_treino, shuffle=True, random_state=10)
            
            #colocar todas as variáveis na mesma escala, usando o conjunto de treinamento para calcular os parâmetros da escala
            ss = StandardScaler()
            ss.fit(X_treino)
            X_treino = ss.transform(X_treino)
            X_teste = ss.transform(X_teste)
            X_val = ss.transform(X_val)

            #escolher o k com o melhor resultado no conjunto de validação e treinar o KNN com o melhor k.
            knn, _, _ = melhorK(range(1,30,2), X_treino, X_val, y_treino, y_val, X_teste, y_teste, None)
        
            y_pred = knn.predict(X_teste)
            y_proba = knn.predict_proba(X_teste)
            
            matriz_info = {
                "fold": foldId,
                "y_true": y_teste.values,
                "y_proba": y_proba,
                "classes": knn.classes_
            }
            
            filename = os.path.join(matrizFoldPath, f"matriz_{foldId + 1}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(matriz_info, f)
            print(f"Probabilidades salvas em {filename}")
            
            os.makedirs(modelosFoldPath, exist_ok=True)
            modelo_filename = os.path.join(modelosFoldPath, f"KNN_model_fold_{foldId + 1}.pkl")
            with open(modelo_filename, "wb") as f_modelo:
                pickle.dump(knn, f_modelo)
            print(f"Modelo do fold {foldId + 1} salvo em {modelo_filename}")
        
        #calcular a acurácia no conjunto de testes desta iteração e salvar na lista.
        
        acuracias.append(f1_score(y_teste, y_pred, average="macro"))
        topKScores.append(top_k_accuracy_score(y_teste, y_proba, k=ka, labels=knn.classes_))
        printResultados(knn, X_teste, y_teste, ka)
    
    return acuracias, topKScores

def main():
    ka = 5
    
    # DATAFRAME SEGMENTADO FOI O UTILIZADO PARA TREINAR OS MODELOS
    
    dataframePath = "../dataframes/dataframeSegmentado.pkl"
    #dataframePath = "../dataframes/dataframeAudioCompleto.pkl"
    #dataframePath = "../dataframes/dataframeAudiosPassaroUnico.pkl"

    if os.path.exists(dataframePath):
        with open(dataframePath, "rb") as readFile:
            df = pickle.load(readFile)
            print("Dataframe carregado com sucesso!")
    else:
        print("Dataframe não encontrado!")
        return

    X = df.drop(columns=["roi_label"]) # x = features
    y = df["roi_label"] # y = passaros

    print("Quantidade de amostras: ", X.shape)
    
    num_especies = y.nunique()
    print("Quantidade de especies: ", num_especies)
    
    acuracias, topKAcuracias = knnCruzado(X, y, ka)
    
    print("\n")
    if(dataframePath == "dataframes/dataframeSegmentado.pkl"):
        print("--TESTE ÁUDIOS SEGMENTADOS--")
    elif(dataframePath == "dataframes/dataframeAudioCompleto.pkl"):
        print("--TESTE ÁUDIOS COMPLETOS--")
    else:
        print("--TESTE ÁUDIOS PÁSSARO ÚNICO--")
        
    print("SCORES KNN: \n")
    
    print("f1-Score Macro:")
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f \n" % (min(acuracias), max(acuracias), npy.mean(acuracias), npy.std(acuracias)))
    
    print(f"Top-K Score (Top-{ka}):")
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f \n" % (min(topKAcuracias), max(topKAcuracias), npy.mean(topKAcuracias), npy.std(topKAcuracias)))

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)
