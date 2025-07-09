import librosa 
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
import os
import numpy as npy
import pickle

audioSourcePath = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\wavs_20241104"
pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"

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
#############

def readCSV(CSV):
    try:
        df = pd.read_csv(CSV, usecols=["soundscape_file", "roi_label", "roi_start", "roi_end", "roi_label_confidence"])
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado")
    except pd.errors.EmptyDataError:
        print("Arquivo Vazio")
#############

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
####################

def process_audio(index, line, lastAudioDict, results):
    audioPath = line["soundscape_file"]
    roiLabel = line["roi_label"]
    startTime = line["roi_start"]
    endTime = line["roi_end"]
    confidence = line["roi_label_confidence"]

    if roiLabel == "NOT_IDENTIFIED" or confidence == "uncertain":
        print(f"linha {index}: Espécie incerta")
        return None

    cutId = lastAudioDict.get(audioPath, -1) + 1
    lastAudioDict[audioPath] = cutId
    
    audio = os.path.join(audioSourcePath, audioPath)

    segmentedAudio, sr = cutAudio(audio, startTime, endTime)
    
    if segmentedAudio is None:
        return None

    centroid, contrast, flatness, rolloff, zeroCrossRate, rms, mfcc = getFeatures(segmentedAudio, sr)
    textures = {
        "roi_label": roiLabel,
        "centroid": centroid,
        "contrast": contrast,
        "flatness": flatness,
        "rolloff": rolloff,
        "zeroCrossRate": zeroCrossRate,
        "rms": rms,
        "mfcc": mfcc.tolist()
    }
    
    results.append([
        textures["roi_label"], textures["centroid"], textures["contrast"],
        textures["flatness"], textures["rolloff"], textures["zeroCrossRate"],
        textures["rms"]] + textures["mfcc"]
    )
    
    print(f"linha{index}: audio = {audioPath}, timeIni = {startTime}, timeFim = {endTime}")

####################

def main():
    df = readCSV(pathCSV)
    
    data = []
    
    #print(df[["roi_start", "roi_end"]].isna().sum())
    
    with Manager() as manager:
        lastAudioDict = manager.dict()

        #df_limite = df.iloc[0:101]

        Parallel(n_jobs=4)(
            delayed(process_audio)(index, line, lastAudioDict) for index, line in df.iterrows()
        )

    columns = ["roi_label", "centroid", "contrast", "flatness", "rolloff", 
            "zeroCrossRate", "rms"] + [f"mfcc_{i}" for i in range(20)]
    dfCut = pd.DataFrame(data, columns=columns) #cria um dataframe pandas

    with open("../dataframes/dataframeSegmentado.pkl", "wb") as file:
        pickle.dump(dfCut, file) #salva as features normalizadas num pickle
    
    #print(dfCut.head())

#############

if __name__ == '__main__':
    main()