import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score

# ============================================
# Configurações iniciais
# ============================================

DATA_VERSIONS = ["v1_media", "v2_media_std", "v3_media_std_freq", "v4_novas_features"]
TYPES = ["segmentado", "completo"]
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

# ============================================
# Funções utilitárias
# ============================================

def carregar_objeto(caminho):
    with open(caminho, "rb") as f:
        return pickle.load(f)

def calcular_metricas(y_true, y_proba, classes, k=TOP_K):
    """
    Calcula F1 macro e Top-k accuracy.
    Filtra y_true e y_proba para manter apenas classes presentes em `classes`.
    """
    # Filtrar amostras cujo rótulo não está em classes (rótulos desconhecidos pelo modelo)
    mask = np.isin(y_true, classes)
    y_true_f = y_true[mask]
    y_proba_f = y_proba[mask]

    if len(y_true_f) == 0:
        return 0.0, 0.0   # fold vazio, retorna 0

    # Predição hard
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
    """Carrega um arquivo de matriz e retorna (f1, topk) ou None se der erro."""
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

def main():
    # Seleção da versão
    print("Selecione a versão do dataset:")
    for i, v in enumerate(DATA_VERSIONS, 1):
        print(f"{i} - {v}")
    idx_v = int(input("Digite o número da versão: ").strip()) - 1
    if idx_v < 0 or idx_v >= len(DATA_VERSIONS):
        print("Versão inválida!")
        return
    version = DATA_VERSIONS[idx_v]

    # Seleção do tipo
    print("\nSelecione o tipo de dataset:")
    for i, t in enumerate(TYPES, 1):
        print(f"{i} - {t.capitalize()}")
    idx_t = int(input("Digite o número do tipo: ").strip()) - 1
    if idx_t < 0 or idx_t >= len(TYPES):
        print("Tipo inválido!")
        return
    tipo = TYPES[idx_t]

    print(f"\n=== Resultados para versão '{version}' - tipo '{tipo}' ===")

    # Para cada classificador, carregar métricas
    resultados_por_classificador = {}

    for nome_clf, sufixo in CLASSIFIERS.items():
        print(f"\n--- Classificador: {nome_clf} ---")
        resultados_por_classificador[nome_clf] = {}

        for n_splits in N_SPLITS:
            # Construir caminho base das matrizes
            base_matrizes = os.path.join(nome_clf, version, f"matrizesProba_{sufixo}_treino{tipo.capitalize()}", f"{n_splits}fold")
            
            if not os.path.exists(base_matrizes):
                print(f"  {n_splits}-fold: diretório não encontrado ({base_matrizes})")
                resultados_por_classificador[nome_clf][n_splits] = None
                continue

            f1_list = []
            topk_list = []

            for fold_id in range(1, n_splits + 1):
                caminho_matriz = os.path.join(base_matrizes, f"matriz_{fold_id}.pkl")
                if not os.path.exists(caminho_matriz):
                    print(f"  Fold {fold_id}: matriz não encontrada")
                    continue

                metricas = carregar_metricas_fold(caminho_matriz)
                if metricas is not None:
                    f1, topk = metricas
                    f1_list.append(f1)
                    topk_list.append(topk)

            if len(f1_list) == 0:
                print(f"  {n_splits}-fold: nenhuma matriz válida encontrada")
                resultados_por_classificador[nome_clf][n_splits] = None
            else:
                f1_mean = np.mean(f1_list)
                f1_std = np.std(f1_list)
                topk_mean = np.mean(topk_list)
                topk_std = np.std(topk_list)
                resultados_por_classificador[nome_clf][n_splits] = {
                    "f1_mean": f1_mean, "f1_std": f1_std,
                    "topk_mean": topk_mean, "topk_std": topk_std
                }
                print(f"  {n_splits}-fold: F1={f1_mean:.4f} ± {f1_std:.4f} | Top-{TOP_K}={topk_mean:.4f} ± {topk_std:.4f}")

    # Exibição final em formato de tabela
    print("\n\n=== Tabela Resumo ===")
    print(f"{'Classificador':<15} {'5-Fold F1':<18} {'5-Fold Top-5':<18} {'10-Fold F1':<18} {'10-Fold Top-5':<18}")
    print("-" * 80)

    for nome_clf in resultados_por_classificador:
        res5 = resultados_por_classificador[nome_clf].get(5)
        res10 = resultados_por_classificador[nome_clf].get(10)

        def fmt(r):
            if r is None:
                return "N/D"
            return f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}"

        def fmt_top(r):
            if r is None:
                return "N/D"
            return f"{r['topk_mean']:.4f}±{r['topk_std']:.4f}"

        print(f"{nome_clf:<15} {fmt(res5):<18} {fmt_top(res5):<18} {fmt(res10):<18} {fmt_top(res10):<18}")

if __name__ == "__main__":
    main()