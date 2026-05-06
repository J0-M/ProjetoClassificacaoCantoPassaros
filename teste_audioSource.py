import pickle
import pandas as pd

############################################
# CONFIG
############################################

DATA_VERSION = "v4_novas_features"
TIPO = "segmentado"

############################################

def carregar(path):
    with open(path, "rb") as f:
        return pickle.load(f)

############################################

def analisar_audio_source(df):

    print("\n=== ANÁLISE DO audioSource ===\n")

    col = df["audioSource"].astype(str)

    ########################################
    # 1. Estatísticas básicas
    ########################################

    print("Total de linhas:", len(col))
    print("Valores únicos (bruto):", col.nunique())

    ########################################
    # 2. Detectar espaços invisíveis
    ########################################

    col_strip = col.str.strip()

    diff_strip = col != col_strip
    print("\n[Espaços extras]")
    print("Linhas afetadas:", diff_strip.sum())

    if diff_strip.sum() > 0:
        print("\nExemplos com espaços:")
        exemplos = pd.DataFrame({
            "original": col[diff_strip].head(10),
            "strip": col_strip[diff_strip].head(10)
        })
        print(exemplos)

    ########################################
    # 3. Detectar diferença de case
    ########################################

    col_lower = col.str.lower()

    diff_case = col_strip != col_lower
    print("\n[Diferença de maiúsculas/minúsculas]")
    print("Linhas afetadas:", diff_case.sum())

    if diff_case.sum() > 0:
        exemplos = pd.DataFrame({
            "original": col_strip[diff_case].head(10),
            "lower": col_lower[diff_case].head(10)
        })
        print(exemplos)

    ########################################
    # 4. Comparar cardinalidade
    ########################################

    print("\n[Impacto da normalização]")

    print("Únicos original:", col.nunique())
    print("Únicos strip:", col_strip.nunique())
    print("Únicos lower:", col_lower.nunique())
    print("Únicos strip+lower:", col_strip.str.lower().nunique())

    ########################################
    # 5. Detectar colisões (mesmo após normalizar)
    ########################################

    col_norm = col.str.strip().str.lower()

    df_temp = pd.DataFrame({
        "original": col,
        "normalizado": col_norm
    })

    duplicados = df_temp.groupby("normalizado")["original"].nunique()
    duplicados = duplicados[duplicados > 1]

    print("\n[Colisões após normalização]")
    print("Grupos com múltiplas variações:", len(duplicados))

    if len(duplicados) > 0:
        print("\nExemplo de colisões:")
        exemplos = duplicados.head(5).index

        for e in exemplos:
            print(f"\nNormalizado: {e}")
            print(df_temp[df_temp["normalizado"] == e]["original"].unique())

############################################

def main():

    df_path = f"dataframes/{DATA_VERSION}/dataframeSegmentado.pkl"

    df = carregar(df_path)

    df = df.dropna(subset=["roi_label"]).copy()

    analisar_audio_source(df)

############################################

if __name__ == "__main__":
    main()