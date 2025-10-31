import os
import numpy as npy
import pickle

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, top_k_accuracy_score

from datetime import datetime

def melhorK(ks, X_treino, X_val, y_treino, y_val, X_teste, y_teste, ka):
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
    print("Acurácia no teste: %.2f" % (f1_score(y_teste, pred, average="macro")))
    
    return knn, melhor_k, melhor_val

def printResultados(knn, X_test_scaled, y_test, ka):
    y_proba = knn.predict_proba(X_test_scaled)
    y_pred = knn.predict(X_test_scaled)

    f1 = f1_score(y_test, y_pred, average="macro")
    
    mask = y_test.isin(knn.classes_)
    y_test_filtrado = y_test[mask]
    y_proba_filtrado = y_proba[mask.values]
    classes_presentes = npy.intersect1d(knn.classes_, npy.unique(y_test_filtrado))
    
    idxs = [npy.where(knn.classes_ == c)[0][0] for c in classes_presentes]
    y_proba_filtrado = y_proba_filtrado[:, idxs]
    
    topk_acc = top_k_accuracy_score(y_test_filtrado, y_proba_filtrado, k=ka, labels=classes_presentes)

    print(f"F1-score do KNN: {f1:.2f}")
    print(f"Top-{ka} Accuracy: {topk_acc:.2f}")
    
def knnCruzado(X, y, ka, groups, dataset_type):
    k_vias = 10
    
    if dataset_type == "segmentado":
        matrizFoldPath = "matrizesProba_knn_treinoSegmentado"
        modelosFoldPath = "modelos_knn_treinoSegmentado"
        path_folds = "folds_audiosSegmentados_knn"
    elif dataset_type == "completo":
        matrizFoldPath = "matrizesProba_knn_treinoCompleto"
        modelosFoldPath = "modelos_knn_treinoCompleto"
        path_folds = "folds_audiosCompletos_knn"
    elif dataset_type == "passaro_unico":
        matrizFoldPath = "matrizesProba_knn_treinoPassaroUnico"
        modelosFoldPath = "modelos_knn_treinoPassaroUnico"
        path_folds = "folds_audiosPassaroUnico_knn"
    else:
        raise ValueError("Tipo de dataset não reconhecido")
    
    skf = StratifiedGroupKFold(n_splits=k_vias, shuffle=True, random_state=10)
    
    acuracias = []
    topKScores = []
    
    counts = y.value_counts()
    classes_validas = counts[counts >= k_vias].index
    filtro = y.isin(classes_validas)
    X = X[filtro]
    y = y[filtro]
    groups = groups[filtro]

    os.makedirs(matrizFoldPath, exist_ok=True)
    os.makedirs(modelosFoldPath, exist_ok=True)
    os.makedirs(path_folds, exist_ok=True)
        
    for foldId, (idx_treino, idx_teste) in enumerate(skf.split(X, y, groups)):
        
        sources_train = set(groups.iloc[idx_treino])
        sources_test = set(groups.iloc[idx_teste])
        intersec = sources_train.intersection(sources_test)
        assert len(intersec) == 0, f"Vazamento detectado em fold {foldId + 1} nos audioSource: {intersec}"
        
        matriz_filename = os.path.join(matrizFoldPath, f"matriz_{foldId + 1}.pkl")
        modelo_filename = os.path.join(modelosFoldPath, f"KNN_model_fold_{foldId + 1}.pkl")
        
        X_treino = X.iloc[idx_treino]
        y_treino = y.iloc[idx_treino]
        X_teste = X.iloc[idx_teste]
        y_teste = y.iloc[idx_teste]
        
        fold_archive_X_treino = os.path.join(path_folds, f"X_treino_fold_{foldId + 1}.pkl")
        fold_archive_y_treino = os.path.join(path_folds, f"y_treino_fold_{foldId + 1}.pkl")
        fold_archive_X_teste = os.path.join(path_folds, f"X_teste_fold_{foldId + 1}.pkl")
        fold_archive_y_teste = os.path.join(path_folds, f"y_teste_fold_{foldId + 1}.pkl")
            
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
        
        if os.path.exists(modelo_filename):
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
            
            counts_treino = y_treino.value_counts()
            classes_validas_treino = counts_treino[counts_treino >= 2].index
            filtro_treino = y_treino.isin(classes_validas_treino)

            X_treino = X_treino[filtro_treino]
            y_treino = y_treino[filtro_treino]
            
            X_treino, X_val, y_treino, y_val = train_test_split(
                X_treino, y_treino, test_size=0.2, stratify=y_treino, shuffle=True, random_state=10)

            ss = StandardScaler()
            ss.fit(X_treino)
            X_treino = ss.transform(X_treino)
            X_teste = ss.transform(X_teste)
            X_val = ss.transform(X_val)

            knn, _, _ = melhorK(range(1,30,2), X_treino, X_val, y_treino, y_val, X_teste, y_teste, None)
        
            y_pred = knn.predict(X_teste)
            y_proba = knn.predict_proba(X_teste)
            
            matriz_info = {
                "fold": foldId,
                "y_true": y_teste.values,
                "y_proba": y_proba,
                "classes": knn.classes_
            }
            
            with open(matriz_filename, "wb") as f:
                pickle.dump(matriz_info, f)
            print(f"Probabilidades salvas em {matriz_filename}")
            
            with open(modelo_filename, "wb") as f_modelo:
                pickle.dump(knn, f_modelo)
            print(f"Modelo do fold {foldId + 1} salvo em {modelo_filename}")

        acuracias.append(f1_score(y_teste, y_pred, average="macro"))
        
        desconhecidas = (~y_teste.isin(knn.classes_)).sum()
        print(f"  Amostras com classe 'não vista no treino': {desconhecidas}/{len(y_teste)}")
        
        classes_treino = set(knn.classes_)
        mask = y_teste.isin(classes_treino)
        
        y_teste_filtrado = y_teste[mask]
        y_proba_filtrado = y_proba[mask.values]
        
        classes_presentes = npy.intersect1d(knn.classes_, npy.unique(y_teste_filtrado))
        idxs = [npy.where(knn.classes_ == c)[0][0] for c in classes_presentes]
        y_proba_filtrado = y_proba_filtrado[:, idxs]
        
        topKScores.append(top_k_accuracy_score(y_teste_filtrado, y_proba_filtrado, k=ka, labels=classes_presentes))
        printResultados(knn, X_teste, y_teste, ka)
    
    return acuracias, topKScores

def main():
    ka = 5  # Hiperparâmetro do Top-K
    
    # Seleção do dataset
    print("Selecione o tipo de dataset:")
    print("1 - Segmentado")
    print("2 - Completo")
    print("3 - Pássaro Único")
    
    choice = input("Digite sua escolha (1-3): ").strip()
    
    if choice == "1":
        dataframePath = "../dataframes/dataframeSegmentado.pkl"
        dataset_type = "segmentado"
        
    elif choice == "2":
        dataframePath = "../dataframes/dataframeAudioCompleto.pkl"
        dataset_type = "completo"
        
    elif choice == "3":
        dataframePath = "../dataframes/dataframeAudiosPassaroUnico.pkl"
        dataset_type = "passaro_unico"
        
    else:
        print("Escolha inválida!")
        return

    if os.path.exists(dataframePath):
        with open(dataframePath, "rb") as readFile:
            df = pickle.load(readFile)
            print("Dataframe carregado com sucesso!")
    else:
        print("Dataframe não encontrado!")
        return

    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]
    groups = df["audioSource"]

    print("Quantidade de amostras: ", X.shape)
    print("Quantidade de especies: ", y.nunique())
    
    acuracias, topKAcuracias = knnCruzado(X, y, ka, groups, dataset_type)
    
    print("\n")
    if dataset_type == "segmentado":
        print("--TESTE ÁUDIOS SEGMENTADOS--")
    elif dataset_type == "completo":
        print("--TESTE ÁUDIOS COMPLETOS--")
    else:
        print("--TESTE ÁUDIOS PÁSSARO ÚNICO--")
        
    print("SCORES KNN: \n")
    
    print("f1-Score Macro:")
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f \n" % (
        min(acuracias), max(acuracias), npy.mean(acuracias), npy.std(acuracias)))
    
    print(f"Top-K Score (Top-{ka}):")
    print("min: %.2f, max: %.2f, avg +- std: %.2f+-%.2f \n" % (
        min(topKAcuracias), max(topKAcuracias), npy.mean(topKAcuracias), npy.std(topKAcuracias)))

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)