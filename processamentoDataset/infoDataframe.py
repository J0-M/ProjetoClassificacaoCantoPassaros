import pickle
import pandas as pd
import numpy as npy
from sklearn.preprocessing import LabelEncoder

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
    pickle_path = "../dataframes/v2_media_std/dataframeSegmentado.pkl" 
    
    dataframe = loadDataframe(pickle_path)
    
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

if __name__ == '__main__':
    main()