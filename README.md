# **Classificação de Cantos de Pássaros**
Este repositório contém o código-fonte e a metodologia para a classificação automatizada de espécies de pássaros utilizando características acústicas extraídas de seus cantos. O projeto compara o desempenho de diferentes algoritmos de aprendizado de máquina (SVM, KNN, XGBoost e K-Means), avaliando o impacto de diferentes versões de features e estratégias de balanceamento de classes. Com o uso de ```random_state``` é possível manter consistência entre os resultados e o pipeline.

<!--
Este trabalho foi desenvolvido como parte de um projeto científico e está documentado no artigo neste link: *(inserir link para o artigo aqui)*.
--> 

## Estrutura do Projeto
O projeto é modularizado por algoritmos, etapas de processamento e versionamento dos experimentos.

```text
ProjetoClassificacaoCantoPassaros/
│
├── dataframes/               # metadados e tabelas de features
├── folds/                    # folds pré-gerados para validação cruzada
├── processamentoDataset/     # scripts de extração de features
├── SVM/                      # pipeline do Support Vector Machine
│   └── SVM.py
├── KNN/                      # pipeline do K-Nearest Neighbors
│   └── KNN.py
├── XGBoost/                  # pipeline do XGBoost
│   └── XGBoost.py
├── KMeansC_NC/               # pipeline KMeans-NC
│   └── KMeansC_NC.py
└── KMeansC_SVM/              # híbridos K-Means + SVM
│   └── KMeansC_SVM.py
│   └── KMeansD.py
```

## Como executar
### 0. Instalação das dependências e Pré-Requisitos
  1. Instale as bibliotecas
  ```bash
  pip install -r requirements.txt
  ```
    
  2. *Dataset* Base
    Você precisará baixar os dados da base conservaSOM e alocá-los na pasta ```data/```.

  3. Modelo *Perch v2*
    Baixe o modelo ```perch_v2_no_dft.onnx``` ([Link](https://huggingface.co/justinchuby/Perch-onnx/tree/da88bf8c37c0b3068fc706fe509f633b9fbd28c1)) e coloque na pasta ```processamentoDataset/features_perch```.

### 1. Extração de Features
Você deve escolher qual versão de features deseja gerar primeiro. O script lerá os áudios, extrairá as características e salvará um ```dataframeSegmentado.pkl```.

#### Para Features Manuais (ex: ```v4_novas_features```):
  ```bash
  python processamentoDataset/v4_novas_features/dataframeSegmentado.py
  ```

  #### Para Embeddings do Perch:
  ```bash
  python processamentoDataset/features_perch/perch.py
  ```
      
### 2. Geração dos folds
Com o dataframe gerado na etapa anterior, execute o script de geração de folds.
  
  ```bash
  python processamentoDataset/folds/gerar_folds.py
  ```

### 3. Treinamento e Avaliação dos Classificadores
Com as features e os folds gerados, você pode rodar qualquer um dos classificadores (**ex**: *XGBoost*):

  ```bash
  python XGBoost/XGBoost.py
  ```

O script é interativo, necessitando que o usuário informe:
1. A versão do dataset.
2. A estratégia de balanceamento (none, ros, smote).
3. O valor de K para o Top-K Accuracy.

O código carregará os folds, fará a busca de hiperparâmetros, treinará o modelo e salvará as matrizes de probabilidade na pasta respectiva do classificador.
O script salva automaticamente as matrizes de confusão/probabilidade e os modelos treinados em formato .pkl nas pastas específicas, permitindo a análise posterior sem a necessidade de retreinar os modelos.
  
## Funcionamento e Metodologia
O fluxo de execução (explicado abaixo) é padronizado para os classificadores supervisionados, garantindo uma comparação justa.

### 1. **Versões do *Dataset* (Conjunto de Características)**
O código permite selecionar diferentes conjuntos de features acústicas (```v1_media```, ```v2_media_std```, ```v3_media_std_freq```, ```v4_novas_features```, ```v6_perch```), permitindo avaliar qual conjunto é mais informativa para o modelo.

```python
CV_SPLITS = [5, 10]

versoes_validas = ["v1_media", "v2_media_std", "v3_media_std_freq", "v4_novas_features", "v5_novo_filtro", "v6_perch"]

print("Selecione a versão do dataset:")
for i, v in enumerate(versoes_validas, 1):
    print(f"{i} - {v}")

entrada = input("Digite o número da versão desejada: ").strip()

if not entrada.isdigit():
    logging.error("Entrada inválida! Digite um número.")
    exit()

idx = int(entrada) - 1

if idx < 0 or idx >= len(versoes_validas):
    logging.error("Versão inválida!")
    exit()

DATA_VERSION = versoes_validas[idx]
```
  
### 2. **Estratégias de Rebalanceamento**
Como datasets de áudio geralmente possuem distribuição de classes desbalanceada, o pipeline permite escolher entre:

- ```none```: Baseline sem balanceamento.

- ```ros``` (*Random Oversampling*): Duplicação aleatória da classe minoritária.

- ```smote``` (*Synthetic Minority Over-sampling Technique*): Geração de amostras sintéticas.
  - O código possui um fallback  para ROS caso a classe minoritária tenha menos de 5 amostras:
```python
DATA_VERSION = versoes_validas[idx]

estrategias_validas = ["none", "ros", "smote"]

NOMES_ESTRATEGIAS = {
    "none":  "Sem balanceamento (baseline)",
    "ros":   "Random Oversampling",
    "smote": "SMOTE (Synthetic Minority Over-sampling)",
}

print("\nSelecione a estratégia de balanceamento:")
for i, e in enumerate(estrategias_validas, 1):
    print(f"{i} - {e:<6s} ({NOMES_ESTRATEGIAS[e]})")

entrada = input("Digite o número da estratégia desejada: ").strip()

if not entrada.isdigit():
    logging.error("Entrada inválida! Digite um número.")
    exit()

idx = int(entrada) - 1

if idx < 0 or idx >= len(estrategias_validas):
    logging.error("Estratégia inválida!")
    exit()
    
REBALANCE = estrategias_validas[idx]
```
```python
def balancear(X, y, strategy, random_state=10):
    if strategy == "none":
        return X, y

    if strategy == "ros":
        return RandomOverSampler(random_state=random_state).fit_resample(X, y)

    if strategy == "smote":
        min_count = min(Counter(y).values())
        k = min(5, min_count - 1)
        
        if k < 1:
            logging.warning(f"Classe minoritária tem {min_count} amostra(s); SMOTE inviável, usando ROS como fallback.")
            return RandomOverSampler(random_state=random_state).fit_resample(X, y)
        
        return SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)
```

### 3. **Estratificação por grupos**
Para garantir que gravações do mesmo indivíduo ou do mesmo ambiente não estejam tanto no treino quanto no teste utilizamos o ```StratifiedGroupKFold``` agrupando pelas colunas audioSource, com ```GroupShuffleSplit``` sendo usado para separar os dados de treino e validação interna durante a busca de hiperparâmetros, mantendo a integridade dos grupos.

```python
X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]
    groups = df["audioSource"]

    skf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
    [...]
```

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
tr_idx, val_idx = next(gss.split(X_treino, y_treino, groups=fold_dict["groups_train"]))

X_tr, X_val = X_treino.iloc[tr_idx], X_treino.iloc[val_idx]
y_tr, y_val = y_treino.iloc[tr_idx], y_treino.iloc[val_idx]
```

## Resultados 
Os modelos são avaliados utilizando F1-Score Macro para lidar com o desbalanceamento entre as espécies e Top-K Accuracy para verificar se a espécie correta está entre as K mais prováveis.

### Desempenho dos Classificadores
Os resultados podem ser visualizados com o script ```tabela_resultados.py```:
```bash
python etc/tabela_resultados.py
```
Melhores resultados provisórios (```v6_perch - none```)
| Classificador | Fold | F1-Macro ± DP | Top-5 ± DP |
| :--- | :---: | :---: | ---: |
| KNN | 5-Fold | 0,38 ± 0,01 | 0,78 ± 0,03 |
|     | 10-Fold | 0,45 ± 0,03 | 0,79 ± 0,07 |
| SVM | 5-Fold | 0,49 ± 0,00 | 0,91 ± 0,01 |
|     | 10-Fold | 0,60 ± 0,02 | 0,94 ± 0,01 |
| XGBoost | 5-Fold | 0,51 ± 0,01 | 0,94 ± 0,01 |
|     | 10-Fold | 0,65 ± 0,01 | 0,96 ± 0,01 |
| KMeansC_NC | 5-Fold | 0,41 ± 0,02 | 0,84 ± 0,01 |
|     | 10-Fold | 0,48 ± 0,03 | 0,86 ± 0,02 |
| KMeansC_SVM | 5-Fold | 0,43 ± 0,01 | 0,81 ± 0,01 |
|     | 10-Fold | 0,50 ± 0,02 | 0,82 ± 0,02 |
| KMeansD | 5-Fold | 0,37 ± 0,01 | 0,87 ± 0,00 |
|     | 10-Fold | 0,48 ± 0,02 | 0,89 ± 0,01 |

<!--
### Impacto do *Oversampling*
ilustração do impacto do oversampling 
-->

## Autores
 - João Marcos Avancini - [Lattes](https://busca-cv-lattes.cnpq.br/curriculo/4080502078325312)
 - Juliano Henrique Foleis - [Lattes](https://busca-cv-lattes.cnpq.br/curriculo/4404958951175349)
 - Diego Bertolini Gonçalves - [Lattes](https://busca-cv-lattes.cnpq.br/curriculo/2264701855770284)
