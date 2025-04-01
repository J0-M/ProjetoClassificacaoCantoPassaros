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
folderDestinyPath = "C:\\Users\\Pichau\\Desktop\\audiosSegmentados"
folderFeaturesPath = "C:\\Users\\Pichau\\Desktop\\texturaAudios"
pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"

def cutAudio(audio, startTime, endTime):
    audio, sr = librosa.load(audio, sr=22000)

    if startTime >= endTime:
        print(f"Erro: startTime ({startTime}) >= endTime ({endTime}) para {audio}")
        return None, None

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

def process_audio(index, line, lastAudioDict):
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
    outputFileName = f"{audioPath}_{cutId}.wav"
    outputPath = os.path.join(folderDestinyPath, outputFileName)

    segmentedAudio, sr = cutAudio(audio, startTime, endTime)
    
    if segmentedAudio is None:
        return None
    
    sf.write(outputPath, segmentedAudio, sr)

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

    npyFileName = f"{audioPath}_{cutId}_features.npy"
    npyPath = os.path.join(folderFeaturesPath, npyFileName)
    npy.save(npyPath, textures)
    
    print(f"linha{index}: audio = {audioPath}, timeIni = {startTime}, timeFim = {endTime}")
    print(f"Segmento Salvo em {outputPath}")
    print(f"Texturas Salvas em {npyPath}")

####################

def main():
    df = readCSV(pathCSV)
    
    with Manager() as manager:
        lastAudioDict = manager.dict()

        df_limite = df.iloc[0:101]

        Parallel(n_jobs=4)(
            delayed(process_audio)(index, line, lastAudioDict) for index, line in df_limite.iterrows()
        )

    data = []

    for file in os.listdir(folderFeaturesPath): # unifica os arquivos das features .npy em uma única matriz para usar o knn
        if file.endswith(".npy"):
            file_path = os.path.join(folderFeaturesPath, file)
            features = npy.load(file_path, allow_pickle=True).item()
            
            if "roi_label" not in features or pd.isnull(features["roi_label"]):
                print(f"Arquivo {file} tem valor ausente ou incorreto para roi_label")
                continue
            
            row = [features["roi_label"], features["centroid"], features["contrast"], 
                features["flatness"], features["rolloff"], features["zeroCrossRate"], 
                features["rms"]] + features["mfcc"]
            
            data.append(row)

    columns = ["roi_label", "centroid", "contrast", "flatness", "rolloff", 
            "zeroCrossRate", "rms"] + [f"mfcc_{i}" for i in range(20)]
    dfCut = pd.DataFrame(data, columns=columns) #cria um dataframe pandas

    with open("dataframeSegmentado.pkl", "wb") as file:
        pickle.dump(dfCut, file) #salva as features normalizadas num pickle
    
    #print(dfCut.head())

#############

if __name__ == '__main__':
    main()