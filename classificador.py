import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report

#path das features
folderFeaturesPath = "C:\\Users\\Pichau\\Desktop\\texturaAudios"

data = []

for file in os.listdir(folderFeaturesPath): # unifica os arquivos das features .npy em uma única matriz para usar o knn
    if file.endswith(".npy"):
        file_path = os.path.join(folderFeaturesPath, file)
        features = np.load(file_path, allow_pickle=True).item()
        
        if "roi_label" not in features or pd.isnull(features["roi_label"]):
            print(f"Arquivo {file} tem valor ausente ou incorreto para roi_label")
            continue
        
        row = [features["roi_label"], features["centroid"], features["contrast"], 
               features["flatness"], features["rolloff"], features["zeroCrossRate"], 
               features["rms"]] + features["mfcc"]
        
        data.append(row)

columns = ["roi_label", "centroid", "contrast", "flatness", "rolloff", 
           "zeroCrossRate", "rms"] + [f"mfcc_{i}" for i in range(20)]
df = pd.DataFrame(data, columns=columns) #cria um dataframe pandas

with open("dataframe.pkl", "wb") as f:
    pickle.dump(df, f) #salva as features normalizadas num pickle



# linha 13 até 32, botar no arquivo de segmentação e salvar como pickle, depois só salvar no inicio do arquivo do classificador (menos custoso que for)



X = df.drop(columns=["roi_label"]) # x = features
y = df["roi_label"] # y = passaros

#print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100) #80% treino, 20% teste

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # normaliza os dados das features para que o treino não seja enviesado por valores distoantes
X_test_scaled = scaler.transform(X_test)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train_scaled, y_train) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

y_pred = knn.predict(X_test_scaled) #testa o knn com o conjunto 20% teste

f1 = f1_score(y_test, y_pred, average="weighted") #f1 score

print(f"K: {k}")
print(f"F1-score do KNN: {f1:.2f}")
print(classification_report(y_test, y_pred))

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f) #salva knn num pickle

print("Modelo salvo como knn_model.pkl")
