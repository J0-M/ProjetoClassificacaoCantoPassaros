import librosa 
import pandas as pd
from joblib import Parallel, delayed
import os
import numpy as np
import pickle

DEBUG = True

N_FFT = 1024

DATA_VERSION = "v4_novas_features"

audioSourcePath = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\wavs_20241104"
pathCSV = "C:\\Users\\Pichau\\Desktop\\dados_RosaGLM_ConservaSom_20241104\\df_ROI_RosaGLM_ConservaSom_20241104.csv"

def cutAudio(audioPath, startTime, endTime):
    if pd.isna(startTime) or pd.isna(endTime):
        print(f"Erro: startTime ou endTime é NaN para {audioPath}")
        return None, None

    if startTime >= endTime:
        print(f"Erro: startTime ({startTime}) >= endTime ({endTime}) para {audioPath}")
        return None, None
    
    if not os.path.exists(audioPath):
        print(f"Erro: arquivo de áudio não encontrado -> {audioPath}")
        return None, None
    
    
    audio, sr = librosa.load(audioPath, sr=22000)

    timeIni = int(sr * startTime)
    timeEnd = int(sr * endTime)

    segmentedAudio = audio[timeIni:timeEnd]
    
    print("Audio length:", len(segmentedAudio))
    print("Audio energy:", np.mean(segmentedAudio**2))
    
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

def getFeatures(audio, sr, minFreq, maxFreq):
    
    def feature(x):
        if x.size == 0:
            return 0.0, 0.0
        return float(np.nanmean(x)), float(np.nanstd(x))
        #return float(x.mean()), float(x.std())
    
    if len(audio) < sr * 0.1:
        return None
    
    # FEATURES VELHAS
    centroid_mean, centroid_std = feature(librosa.feature.spectral_centroid(y=audio, sr=sr))
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean, contrast_std = float(contrast.mean()), float(contrast.std())
    flatness_mean, flatness_std = feature(librosa.feature.spectral_flatness(y=audio))
    rolloff_mean, rolloff_std = feature(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zero_mean, zero_std = feature(librosa.feature.zero_crossing_rate(y=audio))
    rms_mean, rms_std = feature(librosa.feature.rms(y=audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_mels=40, n_fft=2048)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # FEATURES NOVAS
    
    try:
        f0 = librosa.yin(audio, fmin=50, fmax=8000, frame_length=1024)
        f0 = f0[(f0 > 50) & (f0 < 5000)]
        if f0.size > 0:
            f0_mean, f0_std = float(np.mean(f0)), float(np.std(f0))
        else:
            f0_mean, f0_std = 0.0, 0.0
    except Exception as e:
        print("Erro YIN:", e)
        f0_mean, f0_std = 0.0, 0.0
        
    
    try:
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = np.mean(harmonic**2)
        percussive_energy = np.mean(percussive**2)

        if harmonic_energy + percussive_energy < 1e-8:
            ratio_hp = 0.0
        else:
            ratio_hp = float(harmonic_energy / (percussive_energy + 1e-9))
    except Exception as e:
        print("Erro HPSS:", e)
        ratio_hp = 0.0
    
    
    S = np.abs(librosa.stft(audio, n_fft=N_FFT))
    print("S mean:", np.mean(S))
    
    S_sum = np.sum(S, axis=0, keepdims=True)
    S_norm = S / (S_sum + 1e-9)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-9), axis=0)
    entropy_mean, entropy_std = feature(entropy) # Mede complexidade/desordem do som
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    
    # Frequência dominante (pico médio do espectro)
    spec_mean = np.mean(S, axis=1)
    if spec_mean.size > 0:
        valid = freqs > 300
        spec_mean_valid = spec_mean[valid]
        freqs_valid = freqs[valid]

        dominant_freq = float(freqs_valid[np.argmax(spec_mean_valid)])
    else:
        dominant_freq = 0.0


    return (
        centroid_mean, centroid_std,
        contrast_mean, contrast_std,
        flatness_mean, flatness_std,
        rolloff_mean, rolloff_std,
        zero_mean, zero_std,
        rms_mean, rms_std,
        mfcc_mean, mfcc_std,
        f0_mean, f0_std,
        ratio_hp,
        entropy_mean, entropy_std,
        dominant_freq
    )
    
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


    features = getFeatures(segmentedAudio, sr, minFreq, maxFreq)    
    
    if features is None:
        print(f"Linha {index} ignorada (áudio muito curto)")
        return None
    
    (
        centroid_mean, centroid_std,
        contrast_mean, contrast_std,
        flatness_mean, flatness_std,
        rolloff_mean, rolloff_std,
        zero_mean, zero_std,
        rms_mean, rms_std,
        mfcc_mean, mfcc_std,
        f0_mean, f0_std,
        ratio_hp,
        entropy_mean, entropy_std,
        dominant_freq
    ) = features
    
    features_to_check = [centroid_mean, centroid_std, contrast_mean, contrast_std,
                        flatness_mean, flatness_std, rolloff_mean, rolloff_std,
                        zero_mean, zero_std, rms_mean, rms_std, f0_mean, f0_std,
                        ratio_hp,entropy_mean, entropy_std, dominant_freq]
    
    if (
        any(np.isnan(f) or np.isinf(f) for f in features_to_check)
        or np.isnan(mfcc_mean).any()
        or np.isnan(mfcc_std).any()
    ):
        print(f"Features inválidas (NaN/Inf) para {audioPath}")
        return None
    
    row_features = [
        audioPath,roiLabel,
        float(minFreq),float(maxFreq),float(duration),
        
        centroid_mean,centroid_std,
        contrast_mean,contrast_std,
        flatness_mean,flatness_std,
        rolloff_mean,rolloff_std,
        zero_mean,zero_std,
        rms_mean,rms_std,
        
        f0_mean, f0_std,
        ratio_hp,
        entropy_mean, entropy_std,
        dominant_freq
    ] + mfcc_mean.tolist() + mfcc_std.tolist()
    
    print(f"Linha {index} processada com sucesso")
    
    if DEBUG and index < 20:
        print("\n===== DEBUG FEATURES =====")
        print("centroid:", centroid_mean, centroid_std)
        print("flatness:", flatness_mean, flatness_std)
        print("rolloff:", rolloff_mean, rolloff_std)
        print("rms:", rms_mean, rms_std)
        print("f0:", f0_mean, f0_std)
        print("ratio_hp:", ratio_hp)
        print("entropy:", entropy_mean, entropy_std)
        print("dominant_freq:", dominant_freq)
        print("=========================\n")
    
    return row_features

####################

def main():
    
    print(f"Versão = {DATA_VERSION}")
    
    df = readCSV(pathCSV)
    
    if df is None:
        print("Dataframe não encontrado!")
        return
    
    print("Total de Linhas = ", len(df))
    
    results = Parallel(n_jobs=4, prefer="threads")(
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
        "rms_mean", "rms_std",
        "f0_mean", "f0_std",
        "harmonic_ratio",
        "entropy_mean", "entropy_std",
        "dominant_freq"
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