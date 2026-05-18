import os
import pickle
from collections import Counter, defaultdict

#########################################################
# CONFIGURAÇÃO
#########################################################

versoes_validas = ["v1_media", "v2_media_std", "v3_media_std_freq", "v4_novas_features"]

print("Selecione a versão do dataset:")
for i, v in enumerate(versoes_validas, 1):
    print(f"{i} - {v}")

entrada = input("Digite o número da versão desejada: ").strip()

if not entrada.isdigit():
    print("[ERROR] Entrada inválida! Digite um número.")
    exit()

idx = int(entrada) - 1

if idx < 0 or idx >= len(versoes_validas):
    print("[ERROR] Versão inválida!")
    exit()

DATA_VERSION = versoes_validas[idx]

PATH_MODELOS = f"KMeansC_XGBoost/{DATA_VERSION}/modelos_kmeansc_xgboost_treinoSegmentado" # KMEANSC_XGBOOST

#########################################################
# UTILITÁRIOS
#########################################################

def carregar_objeto(caminho):
    with open(caminho, "rb") as f:
        return pickle.load(f)

#########################################################
# MAIN
#########################################################

def main():

    if not os.path.exists(PATH_MODELOS):
        print("Pasta de modelos não encontrada.")
        return

    arquivos = sorted([
        arq for arq in os.listdir(PATH_MODELOS)
        if arq.endswith(".pkl")
    ])

    if len(arquivos) == 0:
        print("Nenhum modelo encontrado.")
        return

    #####################################################
    # HISTÓRICO
    #####################################################

    historico = []

    print("\n========================================")
    print("INFORMAÇÕES DOS MODELOS")
    print("========================================\n")

    for arquivo in arquivos:

        caminho = os.path.join(PATH_MODELOS, arquivo)

        modelo = carregar_objeto(caminho)

        print(f"\n########## {arquivo} ##########")

        #################################################
        # K
        #################################################

        k = modelo.get("k", None)

        print(f"k: {k}")

        #################################################
        # XGBOOST
        #################################################

        xgb = modelo.get("xgboost", None)

        if xgb is not None:

            params = xgb.get_params()

            principais = {
                "max_depth": params.get("max_depth"),
                "learning_rate": params.get("learning_rate"),
                "subsample": params.get("subsample"),
                "colsample_bytree": params.get("colsample_bytree"),
                "n_estimators": params.get("n_estimators"),
            }

            print("\nParâmetros XGBoost:")

            for chave, valor in principais.items():
                print(f"  {chave}: {valor}")

            historico.append({
                "k": k,
                **principais
            })

        #################################################
        # KMEANS
        #################################################

        kmeans = modelo.get("kmeans", None)

        if kmeans is not None:

            centroides = kmeans["centroides"]

            print(f"\nQtd centroides: {len(centroides)}")
            print(f"Dimensão centroides: {centroides.shape[1]}")

    #####################################################
    # RESUMO GERAL
    #####################################################

    print("\n\n========================================")
    print("RESUMO GERAL")
    print("========================================")

    if len(historico) == 0:
        print("Nenhum histórico encontrado.")
        return

    chaves = [
        "k",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "n_estimators"
    ]

    for chave in chaves:

        valores = [h[chave] for h in historico]

        contador = Counter(valores)

        print(f"\n{chave}:")

        for valor, freq in contador.most_common():

            print(f"  {valor}: {freq} folds")

    #####################################################
    # MELHOR CONFIG MAIS FREQUENTE
    #####################################################

    configs = [
        (
            h["k"],
            h["max_depth"],
            h["learning_rate"],
            h["subsample"],
            h["colsample_bytree"],
            h["n_estimators"]
        )
        for h in historico
    ]

    contador_configs = Counter(configs)

    print("\n========================================")
    print("CONFIGURAÇÕES MAIS FREQUENTES")
    print("========================================\n")

    for config, freq in contador_configs.most_common():

        print(f"{freq} folds -> {config}")

#########################################################

if __name__ == "__main__":
    main()