import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score


DATA_VERSIONS = ["v1_media", "v2_media_std", "v3_media_std_freq", "v4_novas_features", "v5_novo_filtro", "v6_perch"]
TYPES = ["segmentado"]
CLASSIFIERS = {
    "KNN": "knn",
    "SVM": "svm",
    "XGBoost": "xgboost",
    "KMeansC_NC": "kmeansc_nc",
    "KMeansC_SVM": "kmeansc_svm",
    "KMeansD": "kmeansd"
}
N_SPLITS = [5, 10]
TOP_K = 5   # altere se necessário

def carregar_objeto(caminho):
    with open(caminho, "rb") as f:
        return pickle.load(f)

def calcular_metricas(y_true, y_proba, classes, k=TOP_K):
    mask = np.isin(y_true, classes)
    y_true_f = y_true[mask]
    y_proba_f = y_proba[mask]

    if len(y_true_f) == 0:
        return 0.0, 0.0

    y_pred = classes[np.argmax(y_proba_f, axis=1)]

    f1 = f1_score(y_true_f, y_pred, average="macro")

    topk = top_k_accuracy_score(
        y_true_f,
        y_proba_f,
        k=k,
        labels=classes
    )

    return f1, topk

def carregar_metricas_fold(caminho_matriz):
    try:
        matriz = carregar_objeto(caminho_matriz)
        y_true = matriz["y_true"]
        y_proba = matriz["y_proba"]
        classes = matriz["classes"]
        f1, topk = calcular_metricas(y_true, y_proba, classes)
        return f1, topk
    except Exception as e:
        print(f"  Erro ao processar {caminho_matriz}: {e}")
        return None

def coletar_resultados(version, tipo):
    resultados_por_classificador = {}

    for nome_clf, sufixo in CLASSIFIERS.items():
        resultados_por_classificador[nome_clf] = {n: None for n in N_SPLITS}

        for n_splits in N_SPLITS:
            if nome_clf == "KMeansD":
                base_matrizes = os.path.join(
                    "..", "KMeansC_SVM", version,
                    f"matrizesProba_{sufixo}_treino{tipo.capitalize()}",
                    f"{n_splits}fold"
                )
            else:
                base_matrizes = os.path.join(
                    "..", nome_clf, version,
                    f"matrizesProba_{sufixo}_treino{tipo.capitalize()}",
                    f"{n_splits}fold"
                )

            if not os.path.exists(base_matrizes):
                continue  # Mantém como None (N/A)

            f1_list = []
            topk_list = []

            for fold_id in range(1, n_splits + 1):
                caminho_matriz = os.path.join(base_matrizes, f"matriz_{fold_id}.pkl")
                if not os.path.exists(caminho_matriz):
                    continue

                metricas = carregar_metricas_fold(caminho_matriz)
                if metricas is not None:
                    f1, topk = metricas
                    f1_list.append(f1)
                    topk_list.append(topk)

            if len(f1_list) > 0:
                resultados_por_classificador[nome_clf][n_splits] = {
                    "f1_mean": np.mean(f1_list),
                    "f1_std": np.std(f1_list),
                    "topk_mean": np.mean(topk_list),
                    "topk_std": np.std(topk_list),
                }

    return resultados_por_classificador

def imprimir_tabela(resultados_por_classificador, version, tipo):
    print(f"\n#### Versão: `{version}` | Tipo: `{tipo.capitalize()}`\n")
    print("| Classificador | 5-Fold F1 | 5-Fold Top-5 | 10-Fold F1 | 10-Fold Top-5 |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    for nome_clf in CLASSIFIERS:  # Itera sobre todos os classificadores definidos
        res5 = resultados_por_classificador[nome_clf].get(5)
        res10 = resultados_por_classificador[nome_clf].get(10)

        def fmt(r, metric="f1"):
            if r is None:
                return "N/A"
            if metric == "f1":
                return f"{r['f1_mean']:.2f} ± {r['f1_std']:.2f}"
            return f"{r['topk_mean']:.2f} ± {r['topk_std']:.2f}"

        print(f"| {nome_clf} | {fmt(res5, 'f1')} | {fmt(res5, 'topk')} | {fmt(res10, 'f1')} | {fmt(res10, 'topk')} |")
    
    print("\n")

def main():
    for tipo in TYPES:
        for version in DATA_VERSIONS:
            print(f"\n\n{'#' * 80}")
            print(f"### Coletando resultados para versão '{version}' - tipo '{tipo}'")
            print(f"{'#' * 80}\n")

            resultados = coletar_resultados(version, tipo)
            imprimir_tabela(resultados, version, tipo)

if __name__ == "__main__":
    main()