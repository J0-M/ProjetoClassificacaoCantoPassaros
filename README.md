# **Classificação de Cantos de Pássaros**
Este repositório contém o código-fonte e a metodologia para a classificação automatizada de espécies de pássaros utilizando características acústicas extraídas de seus cantos. O projeto compara o desempenho de diferentes algoritmos de aprendizado de máquina (SVM, KNN, XGBoost e K-Means), avaliando o impacto de diferentes versões de features e estratégias de balanceamento de classes.

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
    Você precisará baixar os dados da base conservaSOM (<!-- Link para o conservaSOM -->) e alocá-los na pasta ```data/```.

  3. Modelo *Perch v2*
    Baixe o modelo ```perch_v2_no_dft.onnx``` (<!-- Link para o conservaSOM -->) e coloque na pasta ```processamentoDataset/features_perch```.

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
  
## Funcionamento e Metodologia
O fluxo de execução (explicado abaixo) é padronizado para os classificadores supervisionados, garantindo uma comparação justa.

### 1. **Versões do *Dataset***
O código permite selecionar diferentes conjuntos de features acústicas (```v1_media```, ```v2_media_std```, ```v3_media_std_freq```, ```v4_novas_features```, ```v6_perch```), permitindo avaliar qual conjunto é mais informativa para o modelo.
   <!--
    Incluir código de conjunto aqui 
    -->
  
### 2. **Estratégias de Rebalanceamento**
Como datasets de áudio geralmente possuem distribuição de classes desbalanceada, o pipeline permite escolher entre:

- ```none```: Baseline sem balanceamento.

- ```ros``` (*Random Oversampling*): Duplicação aleatória da classe minoritária.

- ```smote``` (*Synthetic Minority Over-sampling Technique*): Geração de amostras sintéticas.
  - O código possui um fallback  para ROS caso a classe minoritária tenha menos de 5 amostras:
    <!--
    Incluir código de seleção de oversalmpling aqui 
    -->

### 3. **Estratificação por grupos**
  <!-- falar dos folds, StratifiedGroupKFold e GrupShuffleSplit para preservar so grupos-->
