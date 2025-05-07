import os
import numpy as npy
import pandas as pd
import pickle
import itertools

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, get_scorer_names, get_scorer
from joblib import Parallel, delayed

#from utils import do_cv_knn

def selecionar_melhor_svm(Cs, gammas, X_treino : npy.ndarray, X_val : npy.ndarray, 
                          y_treino : npy.ndarray, y_val : npy.ndarray, n_jobs=4):
    
    def treinar_svm(C, gamma, X_treino, X_val, y_treino, y_val):
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val)
        return accuracy_score(y_val, pred)
    
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
    svm = SVC(C=melhor_c, gamma=melhor_gamma)
    svm.fit(npy.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return svm, melhor_comb, melhor_val

#Implementa a validação cruzada para avaliar o desempenho da SVM na base de dados com as instâncias X e as saídas y.
#cv_splits indica o número de partições que devem ser criadas.
#Cs é a lista com os valores C que devem ser avaliados na busca exaustiva de parametros para a SVM.
#gammas s é a lista com os valores gamma que devem ser avaliados na busca exaustiva de parametros para a SVM.
def do_cv_svm(X, y, cv_splits, Cs=[1], gammas=['scale']):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    
    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X.iloc[treino_idx]
        y_treino = y.iloc[treino_idx]

        X_teste = X.iloc[teste_idx]
        y_teste = y.iloc[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        svm, _, _ = selecionar_melhor_svm(Cs, gammas, X_treino, X_val, y_treino, y_val)
        pred = svm.predict(X_teste)
        
        printResultados(svm, X_teste, y_teste)

        #acuracias.append(accuracy_score(y_teste, pred))
        acuracias.append(f1_score(y_teste, pred, average="macro"))
        
    
    return acuracias

def printResultados(svm, X_test_scaled, y_test):
    y_pred = svm.predict(X_test_scaled) #testa o knn com o conjunto 20% teste

    f1 = f1_score(y_test, y_pred, average="weighted") #f1 score
        
    print(f"F1-score do KNN: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    

def main():
    
    dataframePath = "dataframeSegmentado.pkl"

    if os.path.exists(dataframePath):
        with open("dataframeSegmentado.pkl", "rb") as readFile:
            df = pickle.load(readFile)
            print("Dataframe carregado com sucesso!")

    X = df.drop(columns=["roi_label"]) # x = features
    y = df["roi_label"] # y = passaros

    print(X.shape)
    
    cv = 10
    
    counts = y.value_counts()
    classes_validas = counts[counts >= cv].index
    filtro = y.isin(classes_validas)
    X = X[filtro]
    y = y[filtro]
    
    acuracias = do_cv_svm(X, y, cv, Cs=[1, 10, 100, 1000], gammas=['scale', 'auto', 2e-2, 2e-3, 2e-4])
    
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f" % (min(acuracias), max(acuracias), npy.mean(acuracias), npy.std(acuracias)))

    
if __name__ == '__main__':
    main()
