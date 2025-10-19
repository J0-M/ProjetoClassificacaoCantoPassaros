import pickle
import pandas as pd

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
        quantidade_de_audios = len(dataframe["audioSource"].unique())
        
        quantidade_de_especies = dataframe["roi_label"].nunique()

        print(f"Quantidade de áudios: {quantidade_de_audios}")
        print(f"Quantidade de espécies de pássaros: {quantidade_de_especies}")
    else:
        print("Dataframe não carregado corretamente!")

def main():
    pickle_path = "../dataframes/dataframeAudioCompleto.pkl" 
    
    dataframe = loadDataframe(pickle_path)
    
    dataInfo(dataframe)

if __name__ == '__main__':
    main()