import librosa 
import pandas as pd
import os
import numpy as npy
import pickle

from datetime import datetime

DATA_VERSION = "v1_media"

pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"
audioSourcePath = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\wavs_20241104"

def readCSV(CSV):
    try:
        df = pd.read_csv(CSV, usecols=["soundscape_file", "roi_label", "roi_start", "roi_end", "roi_label_confidence"])
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado")
    except pd.errors.EmptyDataError:
        print("Arquivo Vazio")
    
def readAudio(audioPath):
    try:
        audio, sr = librosa.load(audioPath, sr=22000)
        if len(audio) == 0:
            raise ValueError("Áudio Vazio")
        return audio, sr
    except Exception as e:
        print(f'Erro: {e}')
        return None, None

def getFeatures(audio, sr):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(y=audio).mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()
    zeroCrossRate = librosa.feature.zero_crossing_rate(y=audio).mean()
    rms = librosa.feature.rms(y=audio).mean()

    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc = npy.mean(mfcc, axis=1)

    return(centroid, contrast, flatness, rolloff, zeroCrossRate, rms, mfcc)

def getFinalDataframe(dfAudios):
    data = []
    
    grouped = dfAudios.drop_duplicates(subset=["soundscape_file"])
    
    print(grouped.shape)
    
    for index, row in enumerate(grouped.itertuples()):
        
        print(f"Processando linha {index} - audio: {row.soundscape_file}")
        
        audioFileName = row["soundscape_file"]
        audioPath = os.path.join(audioSourcePath, audioFileName)
        
        if not os.path.exists(audioPath):
            print("Arquivo Não Encontrado!")
            return None
        else:
            audio, sr = readAudio(audioPath)
            
            if audio is None:
                continue
            
            try:
                centroid, contrast, flatness, rolloff, zeroCrossRate, rms, mfcc = getFeatures(audio, sr)
            except Exception as e:
                print(f"Erro ao extrair features de {audioFileName}: {e}")
                continue
            
            row_data = [
                audioFileName,
                row.roi_label,
                centroid,
                contrast,
                flatness,
                rolloff,
                zeroCrossRate,
                rms,
            ] + mfcc.tolist()
            
            data.append(row_data)
            
    columns = ["audioSource", "roi_label", "centroid", "contrast", "flatness", "rolloff", 
            "zeroCrossRate", "rms"] + [f"mfcc_{i}" for i in range(20)]
    dataframe = pd.DataFrame(data, columns=columns)
    
    return dataframe
            
def main():
    print(f"Versão = {DATA_VERSION}")
    
    df = readCSV(pathCSV) #ler CSV
    
    df = df[
        (df["roi_label"] != "NOT_IDENTIFIED") 
        & (df["roi_label_confidence"] != "uncertain")] #filtrando os audios sem especie identificada/sem certeza
    
    especiesAudio = df.groupby("soundscape_file")["roi_label"].nunique() #pegando quantas especies tem por audio
    
    audiosValidos = especiesAudio[especiesAudio == 1].index.tolist()#pegando os audios cujo só tem 1 especie
    
    df = df[df["soundscape_file"].isin(audiosValidos)]
    
    dataframeFinal = getFinalDataframe(df)
    
    pasta = f"../../dataframes/{DATA_VERSION}"
    os.makedirs(pasta, exist_ok=True)

    output = os.path.join(pasta, "dataframeAudioCompleto.pkl")

    with open(output, "wb") as file:
        pickle.dump(dataframeFinal, file)

    print(dataframeFinal.head())
    print("Dataframe salvo em:", output)
    
if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)