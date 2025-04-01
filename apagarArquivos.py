import os

#Função só para facilitar a apagar os arquivos de teste

folderDestinyPath = "C:\\Users\\Pichau\\Desktop\\audiosSegmentados"
folderFeaturesPath = "C:\\Users\\Pichau\\Desktop\\texturaAudios"

def clearFolder(folderPath):
    if os.path.exists(folderPath):
        for file in os.listdir(folderPath):
            file_path = os.path.join(folderPath, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Erro ao remover {file_path}: {e}")

def main():
    clearFolder(folderDestinyPath)
    clearFolder(folderFeaturesPath)
    
    print("Arquivos Apagados!")

if __name__ == "__main__":
    main()