import librosa 
import pandas as pd
import os
import numpy as npy
import pickle

from datetime import datetime

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
    
def cutAudio(audio, startTime, endTime):
    if pd.isna(startTime) or pd.isna(endTime):
        print(f"Erro: startTime ou endTime é NaN para {audio}")
        return None, None

    if startTime >= endTime:
        print(f"Erro: startTime ({startTime}) >= endTime ({endTime}) para {audio}")
        return None, None
    
    audio, sr = librosa.load(audio, sr=22000)

    timeIni = int(sr * startTime)
    timeEnd = int(sr * endTime)

    segmentedAudio = audio[timeIni:timeEnd]

    return segmentedAudio, sr

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
    
    print(dfAudios.shape)
    
    for index, row in dfAudios.iterrows():
        audioFileName = row["soundscape_file"]
        audioPath = os.path.join(audioSourcePath, audioFileName)
        startTime = row["roi_start"]
        endTime = row["roi_end"]
        
        audio, sr = cutAudio(audioPath, startTime, endTime)
        
        if not os.path.exists(audioPath):
            print("Arquivo Não Encontrado!")
            return None
        else:
            
            if audio is None:
                continue
            
            print(f"Processando: {audioFileName}")
            centroid, contrast, flatness, rolloff, zeroCrossRate, rms, mfcc = getFeatures(audio, sr)
            
            features = {
                "roi_label": row["roi_label"],
                "centroid": centroid,
                "contrast": contrast,
                "flatness": flatness,
                "rolloff": rolloff,
                "zeroCrossRate": zeroCrossRate,
                "rms": rms,
                "mfcc": mfcc.tolist()
            }
            
            row = [features["roi_label"], features["centroid"], features["contrast"], 
                features["flatness"], features["rolloff"], features["zeroCrossRate"], 
                features["rms"]] + features["mfcc"]
            
            data.append(row)
            
    columns = ["roi_label", "centroid", "contrast", "flatness", "rolloff", 
            "zeroCrossRate", "rms"] + [f"mfcc_{i}" for i in range(20)]
    dataframe = pd.DataFrame(data, columns=columns)
    
    return dataframe
            
def main():
    df = readCSV(pathCSV) #ler CSV
    
    df = df[(df["roi_label"] != "NOT_IDENTIFIED") & (df["roi_label_confidence"] != "uncertain")] #filtrando os audios sem especie identificada/sem certeza
    
    especiesAudio = df.groupby("soundscape_file")["roi_label"].nunique() #pegando quantas especies tem por audio
    
    audiosValidos = especiesAudio[especiesAudio == 1].index.tolist() #pegando os audios cujo só tem 1 especie
    
    df = df[df["soundscape_file"].isin(audiosValidos)]
    
    dataframeFinal = getFinalDataframe(df)
    
    with open("../dataframes/dataframeAudiosPassaroUnico.pkl", "wb") as file:
        pickle.dump(dataframeFinal, file) #salva as features normalizadas num pickle
    
    print(dataframeFinal.head)
    
if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)