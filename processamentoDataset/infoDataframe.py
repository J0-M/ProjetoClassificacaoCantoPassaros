from dataclasses import dataclass
import logging
import pickle
import pandas as pd
import numpy as npy
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

<<<<<<< HEAD
DATA_VERSION = "v4_novas_features"
=======
DATA_VERSION = "v3_media_std_freq"
>>>>>>> 0f16d68d7f2615dda9a5a71f88812ddd79285565

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

        print(f"Quantidade de áudios: {quantidade_de_audios}")
        print(f"Quantidade de espécies de pássaros: {quantidade_de_especies}")
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
    
    print("Classes originais:", sorted(npy.unique(y_encoded)))
    print("Classes válidas (>=10 exemplos):", sorted(classes_validas))
    
    print("Recortes por espécies:\n")
    
    pd.set_option('display.max_rows', None)
    contagem_por_especie = dataframe["roi_label"].value_counts()
    df_contagem = contagem_por_especie.reset_index()
    df_contagem.columns = ["especie", "quantidade"]

    print(df_contagem)
    
    plt.figure(figsize=(10,6))
    plt.hist(contagem_por_especie.values, bins=30, edgecolor="black")

    plt.title("Distribuição de recortes por espécie")
    plt.xlabel("Número de recortes por espécie")
    plt.ylabel("Quantidade de espécies")

    plt.grid(alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()