import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

#path das features
folderFeaturesPath = "C:\\Users\\Pichau\\Desktop\\texturaAudios"

data = []

for file in os.listdir(folderFeaturesPath): # unifica os arquivos das features .npy em uma única matriz para usar o knn
    if file.endswith(".npy"):
        file_path = os.path.join(folderFeaturesPath, file)
        features = np.load(file_path, allow_pickle=True).item()
        
        row = [features["roi_label"], features["centroid"], features["contrast"], 
               features["flatness"], features["rolloff"], features["zeroCrossRate"], 
               features["rms"]] + features["mfcc"]
        
        data.append(row)

columns = ["roi_label", "centroid", "contrast", "flatness", "rolloff", 
           "zeroCrossRate", "rms"] + [f"mfcc_{i}" for i in range(20)]
df = pd.DataFrame(data, columns=columns) #cria um dataframe pandas

X = df.drop(columns=["roi_label"]) # x = features
y = df["roi_label"] # y = passaros

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # normaliza os dados das features para que o treino não seja enviesado por valores distoantes

with open("features_labels.pkl", "wb") as f:
    pickle.dump((X_scaled, y), f) #salva as features normalizadas num pickle

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=20) #8-% treino, 20% teste

knn = KNeighborsClassifier(n_neighbors=5)  # k=5
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test) #testa o knn com o conjunto 20% teste

f1 = f1_score(y_test, y_pred, average="weighted") #f1 score

print(f"F1-score do KNN: {f1:.2f}")

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f) #salav knn num pickle

print("Modelo salvo como knn_model.pkl")
