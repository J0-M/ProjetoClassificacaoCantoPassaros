import librosa 
import pandas as pd
import os
import numpy as npy
import pickle

from datetime import datetime

DATA_VERSION = "v2_media_std"

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
    
    def feature(x):
        return x.mean(), x.std()
    
    centroid_mean, centroid_std = feature(librosa.feature.spectral_centroid(y=audio, sr=sr))
    contrast_mean, contrast_std = feature(librosa.feature.spectral_contrast(y=audio, sr=sr))
    
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

def getFinalDataframe(dfAudios):
    data = []
    
    grouped = dfAudios.drop_duplicates(subset=["soundscape_file"])
    
    print(grouped.shape)
    
    for index, row in enumerate(grouped.itertuples(index=False)):
        
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
                ( 
                    centroid_mean, centroid_std,
                    contrast_mean, contrast_std,
                    flatness_mean, flatness_std,
                    rolloff_mean, rolloff_std,
                    zero_mean, zero_std,
                    rms_mean, rms_std,
                    mfcc_mean, mfcc_std 
                ) = getFeatures(audio, sr)
            except Exception as e:
                print(f"Erro ao extrair features de {audioFileName}: {e}")
                continue
            
            row_data = [
                audioFileName,
                row.roi_label,
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
            
            data.append(row_data)
            
    columns = [
        "audioSource", "roi_label",
        "centroid_mean", "centroid_std",
        "contrast_mean", "contrast_std",
        "flatness_mean", "flatness_std",
        "rolloff_mean", "rolloff_std",
        "zeroCrossRate_mean", "zeroCrossRate_std",
        "rms_mean", "rms_std"
    ] + [f"mfcc_mean_{i}" for i in range(20)] + [f"mfcc_std_{i}" for i in range(20)]
    
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