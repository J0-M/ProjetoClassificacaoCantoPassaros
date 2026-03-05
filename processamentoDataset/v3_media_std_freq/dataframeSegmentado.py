import librosa 
import pandas as pd
from joblib import Parallel, delayed
import os
import numpy as np
import pickle

DATA_VERSION = "v3_media_std_freq"

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
        audio, sr = librosa.load(audio, sr=None)
    except Exception as e:
        print(f"Erro ao carregar o áudio {audio}: {e}")
        return None, None

    timeIni = int(sr * startTime)
    timeEnd = int(sr * endTime)

    segmentedAudio = audio[timeIni:timeEnd]
    
    if len(segmentedAudio) == 0:
        return None, None

    return segmentedAudio, sr
#############

def readCSV(CSV):
    try:
        df = pd.read_csv(CSV, usecols=["soundscape_file", "roi_label", "roi_start", "roi_end", "roi_label_confidence", "roi_min_freq", "roi_max_freq", "roi_duration"])
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
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return(centroid_mean, centroid_std,
            contrast_mean, contrast_std,
            flatness_mean, flatness_std,
            rolloff_mean, rolloff_std,
            zero_mean, zero_std,
            rms_mean, rms_std,
            mfcc_mean, mfcc_std)
    
####################

def process_audio(index, row):
    
    print(f"Processando linha {index} - audio: {row.soundscape_file}")
    
    audioPath = row.soundscape_file
    roiLabel = row.roi_label
    startTime = row.roi_start
    endTime = row.roi_end
    confidence = row.roi_label_confidence
    duration = row.roi_duration
    minFreq = row.roi_min_freq
    maxFreq = row.roi_max_freq

    if roiLabel == "NOT_IDENTIFIED" or confidence == "uncertain": # Muitas linhas sem espécie catalogada ou incertas
        print(f"Linha {index} ignorada (espécie incerta)")
        return None
    
    if pd.isna(minFreq) or pd.isna(maxFreq) or pd.isna(duration): # Linhas com valores inválidos são ignoradas
        print(f"Linha {index} com valores de frequencia inválidos")
        return None
    
    audioFullPath = os.path.join(audioSourcePath, audioPath)

    segmentedAudio, sr = cutAudio(audioFullPath, startTime, endTime)
    
    if segmentedAudio is None: # Erro no corte
        print(f"Linha {index} falhou no corte")
        return None

    try:
        (   
            centroid_mean, centroid_std,
            contrast_mean, contrast_std,
            flatness_mean, flatness_std,
            rolloff_mean, rolloff_std,
            zero_mean, zero_std,
            rms_mean, rms_std,
            mfcc_mean, mfcc_std,
        ) = getFeatures(segmentedAudio, sr)
        
    except Exception as e:
        print(f"Linha {index} erro ao extrair features para {audioPath}: {e}")
        return None
    
    features_to_check = [centroid_mean, centroid_std, contrast_mean, contrast_std,
                        flatness_mean, flatness_std, rolloff_mean, rolloff_std,
                        zero_mean, zero_std, rms_mean, rms_std]
    
    if any(np.isnan(f) or np.isinf(f) for f in features_to_check):
        print(f"Features inválidas (NaN/Inf) para {audioPath}")
        return None
    
    row_features = [
        audioPath,
        roiLabel,
        float(minFreq),
        float(maxFreq),
        float(duration),
        centroid_mean,
        centroid_std,
        contrast_mean,
        contrast_std,
        flatness_mean,
        flatness_std,
        rolloff_mean,
        rolloff_std,
        zero_mean,
        zero_std,
        rms_mean,
        rms_std,
    ] + mfcc_mean.tolist() + mfcc_std.tolist()
    
    print(f"Linha {index} processada com sucesso")
    
    return row_features

####################

def main():
    
    print(f"Versão = {DATA_VERSION}")
    
    df = readCSV(pathCSV)
    
    if df is None:
        print("Dataframe não encontrado!")
        return
    
    print("Total de Linhas = ", len(df))
    
    results = Parallel(n_jobs=4)(
        delayed(process_audio)(i, row)
        for i, row in enumerate(df.itertuples(index=False))
    )
        
    data = [r for r in results if r is not None]
    
    columns = [
        "audioSource", "roi_label",
        "min_freq", "max_freq", "duration",
        "centroid_mean", "centroid_std",
        "contrast_mean", "contrast_std",
        "flatness_mean", "flatness_std",
        "rolloff_mean", "rolloff_std",
        "zeroCrossRate_mean", "zeroCrossRate_std",
        "rms_mean", "rms_std"
    ] + [f"mfcc_mean_{i}" for i in range(20)] + [f"mfcc_std_{i}" for i in range(20)]
    
    dfCut = pd.DataFrame(data, columns=columns) #cria um dataframe pandas

    pasta = f"../../dataframes/{DATA_VERSION}"
    os.makedirs(pasta, exist_ok=True)
    
    pathOutput = os.path.join(pasta, "dataframeSegmentado.pkl")
    
    with open(pathOutput, "wb") as file:
        pickle.dump(dfCut, file) #salva as features normalizadas num pickle
    
    print(dfCut.head())
    print("Dataframe Salvo em: ", pathOutput)

#############

if __name__ == '__main__':
    main()