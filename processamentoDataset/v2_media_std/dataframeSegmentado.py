import librosa 
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Lock
import os
import numpy as npy
import pickle

DATA_VERSION = "v2_media_std"

folderFeaturesPath = "C:\\Users\\Pichau\\Desktop\\texturasAudios"

audioSourcePath = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\wavs_20241104"
pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"

def cutAudio(audio, startTime, endTime):
    if pd.isna(startTime) or pd.isna(endTime):
        print(f"Erro: startTime ou endTime é NaN para {audio}")
        return None, None

    if startTime >= endTime:
        print(f"Erro: startTime ({startTime}) >= endTime ({endTime}) para {audio}")
        return None, None
    
    if not os.path.exists(audio):
        print(f"Erro: arquivo de áudio não encontrado -> {audio}")
        return None, None
    
    try:
        audio, sr = librosa.load(audio, sr=22000)
    except Exception as e:
        print(f"Erro ao carregar o áudio {audio}: {e}")
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
    
    def feature(x):
        return float(x.mean()), float(x.std())
    
    centroid_mean, centroid_std = feature(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean, contrast_std = float(contrast.mean()), float(contrast.std())
    
    flatness_mean, flatness_std = feature(librosa.feature.spectral_flatness(y=audio))
    rolloff_mean, rolloff_std = feature(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zero_mean, zero_std = feature(librosa.feature.zero_crossing_rate(y=audio))
    rms_mean, rms_std = feature(librosa.feature.rms(y=audio))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = npy.mean(mfcc, axis=1)
    mfcc_std = npy.std(mfcc, axis=1)

    return(centroid_mean, centroid_std,
            contrast_mean, contrast_std,
            flatness_mean, flatness_std,
            rolloff_mean, rolloff_std,
            zero_mean, zero_std,
            rms_mean, rms_std,
            mfcc_mean, mfcc_std)
    
####################

def process_audio(index, line, lastAudioDict, lock):
    audioPath = line["soundscape_file"]
    roiLabel = line["roi_label"]
    startTime = line["roi_start"]
    endTime = line["roi_end"]
    confidence = line["roi_label_confidence"]

    if roiLabel == "NOT_IDENTIFIED" or confidence == "uncertain":
        print(f"Linha {index}: Espécie incerta")
        return None

    with lock:
        cutId = lastAudioDict.get(audioPath, -1) + 1
        lastAudioDict[audioPath] = cutId
    
    audio = os.path.join(audioSourcePath, audioPath)

    segmentedAudio, sr = cutAudio(audio, startTime, endTime)
    
    if segmentedAudio is None:
        return None
    
    if len(segmentedAudio) == 0:
        print(f"Áudio segmentado vazio para {audioPath}")
        return None

    try:
        ( centroid_mean, centroid_std,
            contrast_mean, contrast_std,
            flatness_mean, flatness_std,
            rolloff_mean, rolloff_std,
            zero_mean, zero_std,
            rms_mean, rms_std,
            mfcc_mean, mfcc_std 
        ) = getFeatures(segmentedAudio, sr)
    except Exception as e:
        print(f"Erro ao extrair features para {audioPath}: {e}")
        return None
    
    features_to_check = [centroid_mean, centroid_std, contrast_mean, contrast_std,
                        flatness_mean, flatness_std, rolloff_mean, rolloff_std,
                        zero_mean, zero_std, rms_mean, rms_std]
    
    if any(npy.isnan(f) or npy.isinf(f) for f in features_to_check):
        print(f"Features inválidas (NaN/Inf) para {audioPath}")
        return None
    
    textures = {
        "audioSource": audioPath,
        "roi_label": roiLabel,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "contrast_mean": contrast_mean,
        "contrast_std": contrast_std,
        "flatness_mean": flatness_mean,
        "flatness_std": flatness_std,
        "rolloff_mean": rolloff_mean,
        "rolloff_std": rolloff_std,
        "zeroCrossRate_mean": zero_mean,
        "zeroCrossRate_std": zero_std,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist()
    }
    
    npyFileName = f"{audioPath}_{cutId}_features.npy"
    npyPath = os.path.join(folderFeaturesPath, npyFileName)
    
    try:
        npy.save(npyPath, textures)
        print(f"Linha {index}: audio = {audioPath}, timeIni = {startTime}, timeFim = {endTime}")
        print(f"Texturas Salvas em {npyPath}")
    except Exception as e:
        print(f"Erro ao salvar features para {audioPath}: {e}")
        return None
    
    print(f"Linha {index}: audio = {audioPath}, timeIni = {startTime}, timeFim = {endTime}")
    print(f"Texturas Salvas em {npyPath}")

####################

def main():
    df = readCSV(pathCSV)
    
    #print(df[["roi_start", "roi_end"]].isna().sum())
    
    if not os.path.exists(folderFeaturesPath):
        os.makedirs(folderFeaturesPath)
    
    with Manager() as manager:
        lastAudioDict = manager.dict()
        lock = manager.Lock()

        #df_limite = df.iloc[0:101]

        Parallel(n_jobs=4)(
            delayed(process_audio)(index, line, lastAudioDict, lock) for index, line in df.iterrows()
        )
        

    data = []

    for file in os.listdir(folderFeaturesPath): # unifica os arquivos das features .npy em uma única matriz para usar o knn
        if file.endswith(".npy"):
            file_path = os.path.join(folderFeaturesPath, file)
            
            try:
                features = npy.load(file_path, allow_pickle=True).item()
                
                required_keys = [
                        'audioSource', 'roi_label', 'centroid_mean', 'centroid_std', 
                        'contrast_mean', 'contrast_std', 'flatness_mean', 'flatness_std',
                        'rolloff_mean', 'rolloff_std', 'zeroCrossRate_mean', 'zeroCrossRate_std', 
                        'rms_mean', 'rms_std', 'mfcc_mean', 'mfcc_std'
                    ]

                missing_keys = [key for key in required_keys if key not in features]
                if missing_keys:
                    print(f"Arquivo {file} faltando chaves: {missing_keys}")
                    continue
                
                if pd.isnull(features["roi_label"]):
                    print(f"Arquivo {file} tem valor ausente ou incorreto para roi_label")
                    continue
                
                if len(features["mfcc_mean"]) != 20 or len(features["mfcc_std"]) != 20:
                    print(f"Arquivo {file} tem MFCCs com tamanho incorreto")
                    continue
                
                try:
                    row = [
                        features["audioSource"], features["roi_label"], 
                        features["centroid_mean"], features["centroid_std"],
                        features["contrast_mean"], features["contrast_std"],
                        features["flatness_mean"], features["flatness_std"],
                        features["rolloff_mean"], features["rolloff_std"],
                        features["zeroCrossRate_mean"], features["zeroCrossRate_std"],
                        features["rms_mean"], features["rms_std"]
                    ] + features["mfcc_mean"] + features["mfcc_std"]
                        
                    data.append(row)
                        
                except Exception as e:
                    print(f"Erro ao construir linha para {file}: {e}")
                    continue
                
            except Exception as e:
                print(f"Erro ao carregar arquivo {file}: {e}")
                continue
    
    columns = [
        "audioSource", "roi_label",
        "centroid_mean", "centroid_std",
        "contrast_mean", "contrast_std",
        "flatness_mean", "flatness_std",
        "rolloff_mean", "rolloff_std",
        "zeroCrossRate_mean", "zeroCrossRate_std",
        "rms_mean", "rms_std"
    ] + [f"mfcc_mean_{i}" for i in range(20)] + [f"mfcc_std_{i}" for i in range(20)]
    
    dfCut = pd.DataFrame(data, columns=columns) #cria um dataframe pandas

    with open(f"../../dataframes/{DATA_VERSION}/dataframeSegmentado.pkl", "wb") as file:
        pickle.dump(dfCut, file) #salva as features normalizadas num pickle
    
    print(dfCut.head())

#############

if __name__ == '__main__':
    main()