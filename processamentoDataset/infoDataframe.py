from dataclasses import dataclass
import logging
import pickle
import pandas as pd
import numpy as npy
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

versoes_validas = ["v1_media", "v2_media_std", "v3_media_std_freq", "v4_novas_features"]

print("Selecione a versão do dataset:")
for i, v in enumerate(versoes_validas, 1):
    print(f"{i} - {v}")

entrada = input("Digite o número da versão desejada: ").strip()

if not entrada.isdigit():
    logging.error("Entrada inválida! Digite um número.")
    exit()

idx = int(entrada) - 1

if idx < 0 or idx >= len(versoes_validas):
    logging.error("Versão inválida!")
    exit()

DATA_VERSION = versoes_validas[idx]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

@dataclass
class DatasetConfig:
    picklePath: str

    
DATASET_CONFIGS = {
    "segmentado": DatasetConfig(
        picklePath=f"../dataframes/{DATA_VERSION}/dataframeSegmentado.pkl"
    ),
    "completo": DatasetConfig(
        picklePath = f"../dataframes/{DATA_VERSION}/dataframeCompleto.pkl"
    ),
}

def loadDataframe(pickle_path):
    try:
        with open(pickle_path, "rb") as file:
            dataframe = pickle.load(file)
        return dataframe
    except FileNotFoundError:
        print("Arquivo pickle não encontrado!")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo pickle: {e}")
        return None

def dataInfo(dataframe):
    if dataframe is not None:
        quantidade_de_audios = len(dataframe)
        quantidade_de_especies = dataframe["roi_label"].nunique()
        quantidade_de_features = dataframe.shape[1]

        print(f"Quantidade de áudios: {quantidade_de_audios}")
        print(f"Quantidade de espécies de pássaros: {quantidade_de_especies}")
        print(f"Quantidade de colunas (features): {quantidade_de_features}")
    else:
        print("Dataframe não carregado corretamente!")

def main():
    
    print(f"VERSÃO = {DATA_VERSION}")
    
    print("Selecione o tipo de dataset:")
    print("1 - Segmentado")
    print("2 - Completo")
    
    opcoes = {"1": "segmentado", "2": "completo"}
    tipo = opcoes.get(input("Digite sua escolha: ").strip())
    
    config = DATASET_CONFIGS[tipo]
    
    dataframe = loadDataframe(config.picklePath)
    
    dataInfo(dataframe)
    
    y = dataframe["roi_label"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    counts = pd.Series(y_encoded).value_counts()
    classes_validas = counts[counts >= 10].index
    filtro = npy.isin(y_encoded, classes_validas)
    y_encoded = y_encoded[filtro]
    
    #print("Classes originais:", sorted(npy.unique(y_encoded)))
    #print("Classes válidas (>=10 exemplos):", sorted(classes_validas))
    
    print("Recortes por espécies:\n")
    
    pd.set_option('display.max_rows', None)
    contagem_por_especie = dataframe["roi_label"].value_counts()
    df_contagem = contagem_por_especie.reset_index()
    df_contagem.columns = ["especie", "quantidade"]

    print(df_contagem)
    
    print("\n=== INFORMAÇÕES DE DESBALANCEAMENTO ===\n")

    valores = contagem_por_especie.values

    min_amostras = valores.min()
    max_amostras = valores.max()
    media_amostras = valores.mean()
    mediana_amostras = npy.median(valores)
    std_amostras = valores.std()

    razao_desbalanceamento = max_amostras / min_amostras if min_amostras > 0 else npy.inf

    print(f"Total de espécies: {len(valores)}")
    print(f"Total de áudios: {len(dataframe)}")
    print(f"Mínimo de recortes por espécie: {min_amostras}")
    print(f"Máximo de recortes por espécie: {max_amostras}")
    print(f"Média de recortes por espécie: {media_amostras:.2f}")
    print(f"Mediana de recortes por espécie: {mediana_amostras}")
    print(f"Desvio padrão: {std_amostras:.2f}")
    print(f"Razão de desbalanceamento (max/min): {razao_desbalanceamento:.2f}")

    
    plt.figure(figsize=(10,6))
    plt.hist(contagem_por_especie.values, bins=30, edgecolor="black")

    plt.title("Distribuição de recortes por espécie")
    plt.xlabel("Número de recortes por espécie")
    plt.ylabel("Quantidade de espécies")

    plt.grid(alpha=0.3)
    #plt.show()
    
    print(dataframe.head())

if __name__ == '__main__':
    main()