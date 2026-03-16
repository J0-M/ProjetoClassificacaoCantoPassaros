import librosa 
import pandas as pd
from joblib import Parallel, delayed
import os
import numpy as np
import pickle

audioSourcePath = "../../data/raw/wavs_20241104/"
pathCSV = "../../data/raw/df_ROI_RosaGLM_ConservaSom_20241104.csv"

def readCSV(CSV):
    try:
        df = pd.read_csv(CSV, usecols=["soundscape_file", "roi_label", "roi_start", "roi_end", "roi_label_confidence", "roi_min_freq", "roi_max_freq", "roi_duration"])
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