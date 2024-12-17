import librosa 
import pandas as pd
import soundfile as sf
from joblib import Parallel
import os

audioSourcePath = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\wavs_20241104"
folderDestinyPath = "C:\\Users\\Pichau\\Desktop\\audiosSegmentados"
pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"

def cutAudio(audio, startTime, endTime):
    audio, sr = librosa.load(audio, sr=None)

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

def main():
    df = readCSV(pathCSV)
    lastAudio = ""
    cutId = 0

    df_limite = df.iloc[0:18]

    for index, line in df_limite.iterrows():
        audioPath = line["soundscape_file"]
        roiLabel = line["roi_label"]
        startTime = line["roi_start"]
        endTime = line["roi_end"]
        confidence = line["roi_label_confidence"]

        if(roiLabel != "NOT_IDENTIFIED" and confidence != "uncertain"):
            if(audioPath == lastAudio):
                cutId+= 1
            else:
                lastAudio = audioPath
                cutId = 0

            audio = os.path.join(audioSourcePath, audioPath)
            
            outputFileName = f"{audioPath}_{cutId}.wav"
            outputPath = os.path.join(folderDestinyPath, outputFileName)

            segmentedAudio, sr = cutAudio(audio, startTime, endTime)

            sf.write(outputPath, segmentedAudio, sr)

            print(f"linha{index}: audio = {audioPath}, timeIni = {startTime}, timeFim = {endTime}")
            print(f"Segmento Salvo em {outputPath}")
        else:
            print(f"linha{index}: Espécie incerta")

#############
main()
