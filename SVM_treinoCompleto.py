import os
import numpy as npy
import pandas as pd
import pickle
import itertools

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score
from joblib import Parallel, delayed

from datetime import datetime

#from utils import do_cv_knn

def selecionar_melhor_svm(ka, Cs, gammas, X_treino : npy.ndarray, X_val : npy.ndarray, 
                          y_treino : npy.ndarray, y_val : npy.ndarray, n_jobs=4):
    
    def treinar_svm(C, gamma, X_treino, X_val, y_treino, y_val):
        svm = SVC(C=C, gamma=gamma, probability=True)
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val)
        #proba = svm.predict_proba(X_val)
        
        return f1_score(y_val, pred, average="macro")
        #return top_k_accuracy_score(y_val, proba, k=ka)
    
    #gera todas as combinações de parametros C e gamma, de acordo com as listas de valores recebidas por parametro.
    #Na prática faz o produto cartesiano entre Cs e gammas.
    combinacoes_parametros = list(itertools.product(Cs, gammas))
    
    #Treinar modelos com todas as combinações de C e gamma
    acuracias_val = Parallel(n_jobs=n_jobs)(delayed(treinar_svm)
                                       (c, g, X_treino, X_val, y_treino, y_val) for c, g in combinacoes_parametros)       
    
    melhor_val = max(acuracias_val)
    #Encontrar a combinação que levou ao melhor resultado no conjunto de validação
    melhor_comb = combinacoes_parametros[npy.argmax(acuracias_val)]   
    melhor_c = melhor_comb[0]
    melhor_gamma = melhor_comb[1]
    
    #Treinar uma SVM com todos os dados de treino e validação usando a melhor combinação de C e gamma.
    svm = SVC(C=melhor_c, gamma=melhor_gamma, probability=True)
    svm.fit(npy.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return svm, melhor_comb, melhor_val

#Implementa a validação cruzada para avaliar o desempenho da SVM na base de dados com as instâncias X e as saídas y.
#cv_splits indica o número de partições que devem ser criadas.
#Cs é a lista com os valores C que devem ser avaliados na busca exaustiva de parametros para a SVM.
#gammas s é a lista com os valores gamma que devem ser avaliados na busca exaustiva de parametros para a SVM.
def do_cv_svm(X, y, ka, cv_splits, Cs=[1], gammas=['scale']):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    topkScores = []
    
    matrizFoldPath = "matrizesProba_svm"
    if(not os.path.exists(matrizFoldPath)):
        os.makedirs(matrizFoldPath, exist_ok=True)
        
    modelosFoldPath = "modelos_svm_treinoCompleto"
    if(not os.path.exists(modelosFoldPath)):
        os.makedirs(modelosFoldPath, exist_ok=True)
    
    path_folds = "folds_audiosCompletos"
    
    for foldId, (treino_idx, teste_idx) in enumerate(skf.split(X, y)):
        
        matriz_filename = os.path.join(matrizFoldPath, f"matriz_{foldId + 1}.pkl")
        modelo_filename = os.path.join(modelosFoldPath, f"svm_model_fold_{foldId + 1}.pkl")
        
        ss = StandardScaler()
        
        X_treino = X.iloc[treino_idx]
        y_treino = y.iloc[treino_idx]

        X_teste = X.iloc[teste_idx]
        y_teste = y.iloc[teste_idx]
        
        fold_archive_X_treino = os.path.join(path_folds, f"X_treino_fold_{foldId + 1}.pkl")
        fold_archive_y_treino = os.path.join(path_folds, f"y_treino_fold_{foldId + 1}.pkl")
        fold_archive_X_teste = os.path.join(path_folds, f"X_teste_fold_{foldId + 1}.pkl")
        fold_archive_y_teste = os.path.join(path_folds, f"y_teste_fold_{foldId + 1}.pkl")

        if(not os.path.exists(path_folds)):
            os.makedirs(path_folds, exist_ok=True)
            
        with open(fold_archive_X_treino, "wb") as f:
            pickle.dump(X_treino, f)
        print(f"Folds salvos em {fold_archive_X_treino}")
        
        with open(fold_archive_y_treino, "wb") as f:
            pickle.dump(y_treino, f)
        print(f"Folds salvos em {fold_archive_y_treino}")
        
        with open(fold_archive_X_teste, "wb") as f:
            pickle.dump(X_teste, f)
        print(f"Folds salvos em {fold_archive_X_teste}")
        
        with open(fold_archive_y_teste, "wb") as f:
            pickle.dump(y_teste, f)
        print(f"Folds salvos em {fold_archive_y_teste}")
        
        if os.path.exists(modelo_filename):
            print(f"Carregando modelo do fold {foldId + 1}...")
            with open(modelo_filename, "rb") as f_modelo:
                svm = pickle.load(f_modelo)

            ss.fit(X_treino)
            X_teste = ss.transform(X_teste)
            y_pred = svm.predict(X_teste)
            y_proba = svm.predict_proba(X_teste)

            if os.path.exists(matriz_filename):
                print(f"Carregando matriz do fold {foldId + 1}...")
                with open(matriz_filename, "rb") as f:
                    matriz_info = pickle.load(f)
            else:
                matriz_info = {
                    "fold": foldId,
                    "y_true": y_teste.values,
                    "y_proba": y_proba,
                    "classes": svm.classes_
                }
                with open(matriz_filename, "wb") as f:
                    pickle.dump(matriz_info, f)
                print(f"Probabilidades salvas em {matriz_filename}")
        
        else:
            print(f"Criando modelo do fold {foldId + 1}...")

            X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

            ss = StandardScaler()
            ss.fit(X_treino)
            X_treino = ss.transform(X_treino)
            X_teste = ss.transform(X_teste)
            X_val = ss.transform(X_val)

            svm, _, _ = selecionar_melhor_svm(ka, Cs, gammas, X_treino, X_val, y_treino, y_val)
            y_pred = svm.predict(X_teste)
            y_proba = svm.predict_proba(X_teste)
        
        
            matriz_info = {
                "fold": foldId,
                "y_true": y_teste.values,
                "y_proba": y_proba,
                "classes": svm.classes_
            }
            
            filename = os.path.join(matrizFoldPath, f"matriz_{foldId + 1}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(matriz_info, f)
            print(f"Probabilidades salvas em {filename}")
            
            os.makedirs(modelosFoldPath, exist_ok=True)
            modelo_filename = os.path.join(modelosFoldPath, f"svm_model_fold_{foldId + 1}.pkl")
            with open(modelo_filename, "wb") as f_modelo:
                pickle.dump(svm, f_modelo)
            print(f"Modelo do fold {foldId + 1} salvo em {modelo_filename}")
            
            
        f1 = f1_score(y_teste, y_pred, average="macro")
        topk = top_k_accuracy_score(y_teste, y_proba, k=ka, labels=svm.classes_)
        printResultados(svm, X_teste, y_teste, ka)

        #acuracias.append(accuracy_score(y_teste, pred))
        acuracias.append(f1)
        topkScores.append(topk)
    
    return acuracias, topkScores

def printResultados(svm, X_test_scaled, y_test, ka):
    y_proba = svm.predict_proba(X_test_scaled)
    y_pred = svm.predict(X_test_scaled)

    f1 = f1_score(y_test, y_pred, average="macro")
    topk_acc = top_k_accuracy_score(y_test, y_proba, k=ka, labels=svm.classes_)

    print(f"F1-score do SVM: {f1:.2f}")
    print(f"Top-{ka} Accuracy: {topk_acc:.2f}")
    #print(classification_report(y_test, y_pred))

    return topk_acc

def main():
    
    k = 5
    
    # DATAFRAME SEGMENTADO FOI O UTILIZADO PARA TREINAR OS MODELOS
    
    #dataframePath = "dataframes/dataframeSegmentado.pkl"
    dataframePath = "dataframes/dataframeAudioCompleto.pkl"
    #dataframePath = "dataframes/dataframeAudiosPassaroUnico.pkl"

    if os.path.exists(dataframePath):
        with open(dataframePath, "rb") as readFile:
            df = pickle.load(readFile)
            print("Dataframe carregado com sucesso!")
    else:
        print("Dataframe não encontrado!")
        return

    X = df.drop(columns=["roi_label"]) # x = features
    y = df["roi_label"] # y = passaros

    print(X.shape)
    
    cv = 10
    
    counts = y.value_counts()
    classes_validas = counts[counts >= cv].index
    filtro = y.isin(classes_validas)
    X = X[filtro]
    y = y[filtro]
    
    acuracias, topkAcuracias = do_cv_svm(X, y, k, cv, Cs=[1, 10, 100, 1000], gammas=['scale', 'auto', 2e-2, 2e-3, 2e-4])
    
    print("\n")
    if(dataframePath == "dataframes/dataframeSegmentado.pkl"):
        print("--TESTE ÁUDIOS SEGMENTADOS--")
    elif(dataframePath == "dataframes/dataframeAudioCompleto.pkl"):
        print("--TESTE ÁUDIOS COMPLETOS--")
    else:
        print("--TESTE ÁUDIOS PÁSSARO ÚNICO--")
    
    print("SCORES SVM: \n")
    
    print("f1-Score Macro:")
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f" % (min(acuracias), max(acuracias), npy.mean(acuracias), npy.std(acuracias)))
    
    print(f"Top-K Score (Top-{k}):")
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f" % (min(topkAcuracias), max(topkAcuracias), npy.mean(topkAcuracias), npy.std(topkAcuracias)))

    
if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)
