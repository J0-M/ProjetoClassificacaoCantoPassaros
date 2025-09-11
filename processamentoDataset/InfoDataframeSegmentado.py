import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import Manager
import os
import pickle

audioSourcePath = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\wavs_20241104"
pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"
saveDFPath = "../dataframes/InfoDataframeSegmentado.pkl"

def readCSV(CSV):
    try:
        df = pd.read_csv(CSV, usecols=["soundscape_file", "roi_label", "roi_start", "roi_end", "roi_label_confidence"])
        return df
    except FileNotFoundError:
        print("Arquivo não encontrado")
    except pd.errors.EmptyDataError:
        print("Arquivo Vazio")

def process_info(index, line, lastAudioDict):
    audioPath = line["soundscape_file"]
    roiLabel = line["roi_label"]
    startTime = line["roi_start"]
    endTime = line["roi_end"]
    confidence = line["roi_label_confidence"]

    if roiLabel == "NOT_IDENTIFIED" or confidence == "uncertain":
        print(f"linha {index}: Espécie incerta")
        return None

    # Controla o número de cortes por áudio
    cutId = lastAudioDict.get(audioPath, -1) + 1
    lastAudioDict[audioPath] = cutId

    audio = os.path.join(audioSourcePath, audioPath)
    
    if not os.path.exists(audio):
        print(f"Erro: arquivo de áudio não encontrado -> {audio}")
        return None

    if pd.isna(startTime) or pd.isna(endTime) or startTime >= endTime:
        print(f"Erro na linha {index}: startTime = {startTime}, endTime = {endTime}")
        return None

    return {
        "index": index,
        "audioSource": audioPath,
        "roi_label": roiLabel,
        "startTime": startTime,
        "endTime": endTime,
        "confidence": confidence,
        "cutId": cutId
    }

def main():
    df = readCSV(pathCSV)
    if df is None:
        return

    with Manager() as manager:
        lastAudioDict = manager.dict()

        results = Parallel(n_jobs=4)(
            delayed(process_info)(index, row, lastAudioDict) for index, row in df.iterrows()
        )

    # Filtra None
    results = [r for r in results if r is not None]

    df_info = pd.DataFrame(results)
    
    with open(saveDFPath, "wb") as file:
        pickle.dump(df_info, file)

    print(df_info.shape[0])
    print(f"DataFrame salvo em {saveDFPath}")

if __name__ == '__main__':
    main()
