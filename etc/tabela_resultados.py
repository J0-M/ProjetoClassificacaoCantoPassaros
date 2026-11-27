import os
import pickle
import numpy as np
import glob
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
BALANCEAMENTOS = ["none", "ros", "smote"]
N_SPLITS = [5, 10]
TOP_K = 5   # altere se necessário

# Controle para não floodar o terminal com o mesmo erro do Pandas
_erros_ja_impressos = set()

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
    except AttributeError as e:
        if "StringDtype" in str(e):
            if "STRINGDTYPE_ERRO" not in _erros_ja_impressos:
                _erros_ja_impressos.add("STRINGDTYPE_ERRO")
                print("\n  [ERRO DE COMPATIBILIDADE DETECTADO]")
                print("  Os arquivos .pkl foram gerados com uma versão do Pandas diferente da atual.")
                print("  SOLUÇÃO: pip install pandas==2.2.3 (ou a versão listada no seu requirements.txt)\n")
        else:
            print(f"  Erro desconhecido ao processar {caminho_matriz}: {e}")
        return None
    except Exception as e:
        print(f"  Erro ao processar {caminho_matriz}: {e}")
        return None

def coletar_resultados(version, tipo):
    # Estrutura: resultados[clf][balanceamento][n_splits] = métricas
    resultados = {}

    for nome_clf, sufixo in CLASSIFIERS.items():
        resultados[nome_clf] = {bal: {n: None for n in N_SPLITS} for bal in BALANCEAMENTOS + ["default"]}

        for n_splits in N_SPLITS:
            if nome_clf == "KMeansD":
                base_dir = os.path.join("..", "KMeansC_SVM", version)
            else:
                base_dir = os.path.join("..", nome_clf, version)

            # Procura a pasta "matrizesProba..." em qualquer nível de subpasta
            search_pattern = os.path.join(base_dir, "**", f"matrizesProba_{sufixo}_treino{tipo.capitalize()}", f"{n_splits}fold")
            matching_dirs = glob.glob(search_pattern, recursive=True)

            for base_matrizes in matching_dirs:
                # Identifica qual a estratégia de balanceamento baseada no caminho da pasta
                path_normalized = base_matrizes.replace("\\", "/")
                if f"/none/" in path_normalized:
                    bal = "none"
                elif f"/ros/" in path_normalized:
                    bal = "ros"
                elif f"/smote/" in path_normalized:
                    bal = "smote"
                else:
                    bal = "default"

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
                    resultados[nome_clf][bal][n_splits] = {
                        "f1_mean": np.mean(f1_list),
                        "f1_std": np.std(f1_list),
                        "topk_mean": np.mean(topk_list),
                        "topk_std": np.std(topk_list),
                    }

    return resultados

def imprimir_tabela(resultados, version, tipo):
    print(f"\n#### Versão: `{version}` | Tipo: `{tipo.capitalize()}`\n")
    print("| Classificador | Balanceamento | 5-Fold F1 | 5-Fold Top-5 | 10-Fold F1 | 10-Fold Top-5 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for nome_clf in CLASSIFIERS:
        # Itera sobre os balanceamentos e também sobre o "default" (caso exista)
        for bal in BALANCEAMENTOS + ["default"]:
            res5 = resultados[nome_clf][bal].get(5)
            res10 = resultados[nome_clf][bal].get(10)

            # Se for "default" e estiver vazio, não imprime para não poluir
            if bal == "default" and res5 is None and res10 is None:
                continue

            def fmt(r, metric="f1"):
                if r is None:
                    return "N/A"
                if metric == "f1":
                    return f"{r['f1_mean']:.2f} ± {r['f1_std']:.2f}"
                return f"{r['topk_mean']:.2f} ± {r['topk_std']:.2f}"

            # Formata o nome do balanceamento para ficar mais bonito
            bal_formatado = f"`{bal}`" if bal != "default" else "*(padrão)*"

            print(f"| {nome_clf} | {bal_formatado} | {fmt(res5, 'f1')} | {fmt(res5, 'topk')} | {fmt(res10, 'f1')} | {fmt(res10, 'topk')} |")
    
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