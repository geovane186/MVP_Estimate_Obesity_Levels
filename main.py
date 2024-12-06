# 1) Import de bibliotecas

# Configuração para não exibir os warnings
import warnings
warnings.filterwarnings("ignore")

# Imports de bibliotecas de manipulação de dados
import pandas as pd  # Manipulação de dados em tabelas
import numpy as np   # Funções matemáticas e arrays

# Imports para visualização
import matplotlib.pyplot as plt  # Criação de gráficos e visualizações

# Pré-processamento e transformação de dados
from sklearn.compose import ColumnTransformer  # Composição de pré-processamento para pipelines
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer  # Escalonamento e transformação de variáveis
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel  # Seleção de atributos

# Divisão e validação do modelo
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV  # Particionamento dos dados e validação cruzada

# Métricas e avaliação
from sklearn.metrics import accuracy_score  # Avaliação de desempenho do modelo

# Utilização de pipelines
from sklearn.pipeline import Pipeline  # Facilitando a organização do processo de aprendizado de máquina

# Modelos de classificação
from sklearn.linear_model import LogisticRegression  # Regressão logística
from sklearn.tree import DecisionTreeClassifier  # Árvore de decisão
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.svm import SVC  # Máquina de vetor de suporte (SVM)
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier  # Modelos ensemble


# 2) Carga do DataSet por url

# url a importar
url_dados = 'https://raw.githubusercontent.com/geovane186/MVP_Estimate_Obesity_Levels/refs/heads/main/DataSet/ObesityDataSet_raw_and_data_sinthetic.csv'

# Carga do dataset através do csv
obesityDataSet = pd.read_csv(url_dados)

# Verifica o tipo de obesityDataSet
print('Classe do DataSet:',type(obesityDataSet), '\n')

# Exibe as 5 primeiras linhas
print(obesityDataSet.head(), '\n')

# Verificando se existem valores ausentes
print('Check de valores ausentes:\n',obesityDataSet.isnull().sum())

# Verificando os tipos das colunas
print('\n Tipos das colunas:\n',obesityDataSet.dtypes)

# 3) Divisão inicial do dataset (Holdout e Validação Cruzada)

seed = 42 # Semente para reprodutibilidade

testSize = 0.20 # tamanho do conjunto de teste

# Separação em conjuntos de treino e teste
X = obesityDataSet.drop(columns=['NObeyesdad'])
y = obesityDataSet['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=testSize, shuffle=True, random_state=seed, stratify=y) # holdout com estratificação

# Parâmetros e partições da validação cruzada
scoring = 'accuracy'
num_particoes = 10
kfold = StratifiedKFold(n_splits=num_particoes, shuffle=True, random_state=seed) # validação cruzada com estratificação

# 4) Codificação de atributos

np.random.seed(42)

# Codificação personalizada (usando funções auxiliares)
def encode_ordinal(data, columns, mapping_dicts):
    for col, mapping in zip(columns, mapping_dicts):
        data[col] = data[col].map(mapping)
    return data
    
# Mapeamento explícito para variáveis categóricas
ordinal_mappings = [
    {'Female': 0, 'Male': 1}, # Gender
    {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},  # CAEC
    {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}  # CALC
]

# Definir colunas categóricas
ordinal_cols = ['Gender', 'CAEC', 'CALC']
nominal_cols = ['MTRANS']
categorical_simple_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

# Criar transformações
ordinal_transformer = FunctionTransformer(
    encode_ordinal, kw_args={'columns': ordinal_cols, 'mapping_dicts': ordinal_mappings}
)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')
# Para as variáveis categóricas simples (LabelEncoder em cada coluna)
simple_transformer = FunctionTransformer(lambda df: df.apply(lambda col: LabelEncoder().fit_transform(col)))

preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_cols),
        ('nominal', nominal_transformer, nominal_cols),
        ('simple', simple_transformer, categorical_simple_cols)
    ],
    remainder='passthrough'  # Manter outras colunas inalteradas
)

# Codificação da variável target
target_mapping = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}

# Codificar a variável target
y_train = y_train.map(target_mapping)
y_test = y_test.map(target_mapping)

# 5) Execução de Modelos para Análise
np.random.seed(42) # Definindo uma semente global

# Listas para armazenar os armazenar os pipelines e os resultados para todas as visões do dataset
pipelines = []
results = []
names = []

# Definindo o classificador base para o BaggingClassifier
base = DecisionTreeClassifier()

# Criando os modelos para o VotingClassifier
bases = []
bases.append(('LR', LogisticRegression(max_iter=200)))
bases.append(('CART', DecisionTreeClassifier()))
bases.append(('SVM', SVC()))

# Criando os elementos do pipeline
models = [
    ('LR', LogisticRegression(max_iter=200)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC()),
    ('Bag', BaggingClassifier(estimator=base)),
    ('RF', RandomForestClassifier()),
    ('ET', ExtraTreesClassifier()),
    ('Ada', AdaBoostClassifier()),
    ('GB', GradientBoostingClassifier()),
    ('Voting', VotingClassifier(bases))
]

# Definindo os pré-processadores
scalers = [
    ('StandardScaler', StandardScaler()), # Padronizador
    ('MinMaxScaler', MinMaxScaler()) # Normalizador
]

# Técnicas de seleção de atributos
selection_methods = [
    ('select_kbest', SelectKBest(score_func=f_classif, k=10)),
    ('rfe', RFE(LogisticRegression(max_iter=200), n_features_to_select=10)),
    ('selectET', SelectFromModel(ExtraTreesClassifier(n_estimators=100), threshold='mean'))
]

# Criando pipelines dinamicamente

# Dataset original
for model_name, model in models:
    pipelines.append((
        f'{model_name}',
        Pipeline(steps=[('preprocessing', preprocessor),
                        (model_name, model)])
        ))

# Dataset original com Feature Selection
for model_name, model in models:
    for selection_name, selection in selection_methods:
        pipelines.append((
            f'{model_name}-{selection_name}',
            Pipeline(steps=[('preprocessing', preprocessor),
                            (selection_name, selection),
                            (model_name, model)])
        ))

# Dataset padronizado e normalizado
for model_name, model in models:
    for scaler_name, scaler in scalers:
        pipelines.append((
            f'{model_name}-{scaler_name}',
            Pipeline(steps=[('preprocessing', preprocessor), 
                            (scaler_name, scaler), 
                            (model_name, model)])
        ))

# Dataset padronizado e normalizado com Feature Selection
for model_name, model in models:
    for scaler_name, scaler in scalers:
        for selection_name, selection in selection_methods:
            # Definindo o nome do pipeline com base no modelo, pré-processador e técnica de seleção
            pipeline_name = f'{model_name}-{scaler_name}-{selection_name}'
            pipelines.append((
                pipeline_name,
                Pipeline(steps=[('preprocessing', preprocessor),
                                (scaler_name, scaler),
                                (selection_name, selection),
                                (model_name, model)])
            ))

# Executando os pipelines
for name, model in pipelines:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std()) # formatando para 3 casas decimais
    print(msg)

# Boxplot de comparação dos modelos
fig = plt.figure(figsize=(25,6))
fig.suptitle('Comparação dos Modelos - Dataset orginal, padronizado e normalizado') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names, rotation=90)
plt.show()

# 6) Otimização de Hiperparametros
# Tuning do GradientBoosting
np.random.seed(42) # definindo uma semente global para este bloco

pipelines = []

# Instanciando o padronizador
standard_scaler = ('StandardScaler', StandardScaler())

# Instanciando o SelectKBest
best_var = SelectKBest(score_func=f_classif, k=10)

# Instanciando o classificador
gbc = GradientBoostingClassifier()
gradient_boosting = ('GB', gbc)

pipelines.append(('GB-select_kbest', Pipeline(steps=[('preprocessing', preprocessor), ('select_kbest', best_var), gradient_boosting]))) # GB com DataSet original com Feature Selection
pipelines.append(('GB-StandardScaler-select_kbest', Pipeline(steps=[('preprocessing', preprocessor),standard_scaler, ('select_kbest', best_var), gradient_boosting]))) # GB com DataSet padronizado com Feature Selection

param_distributions = {
    'GB__n_estimators': [100, 200, 500, 1000],
    'GB__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5],
    'GB__subsample': [0.8, 0.9, 1.0],
    'GB__max_depth': [3, 5, 10, 20],
    'GB__min_samples_split': [2, 5, 10],
    'GB__min_samples_leaf': [1, 2, 4],
    'GB__max_features': [None, 'sqrt', 'log2']
}

best_accuracy = 0
best_model = None
best_params = None


# Prepara e executa o RandomizedSearchCV
for name, model in pipelines:    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=10, scoring=scoring, cv=kfold, n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)
    # imprime a melhor configuração
    print("Sem tratamento de missings: %s - Melhor: %f usando %s" % (name, random_search.best_score_, random_search.best_params_)) 
    
    # Obtendo o melhor modelo da busca
    current_best_model = random_search.best_estimator_
    current_best_accuracy = random_search.best_score_
    current_best_params = random_search.best_params_

    # Verificando se o modelo atual tem a melhor acurácia
    if current_best_accuracy > best_accuracy:
        best_accuracy = current_best_accuracy
        best_model = current_best_model
        best_params = current_best_params

# 7) Avaliação do modelo com o conjunto de testes
print(best_accuracy)
print(best_model)
print(best_params)

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

# Calculando a acurácia no conjunto de teste
test_accuracy = accuracy_score(y_test, predictions)

print(test_accuracy)

# 8) Preparação do modelo com TODO o dataset
best_model.fit(X, y)

# 9) Simulação com novos dados não vistos

# Novos dados - Removidos do DataSet Original antes do carregamento.
data = {
    'Gender': ['Female', 'Female', 'Female'],
    'Age': [23.0, 16.0, 24.0],
    'Height': [1.6, 1.61, 1.6],
    'Weight': [52.0, 65.0, 100.5],
    'family_history_with_overweight': ['no', 'yes', 'yes'],
    'FAVC': ['yes', 'yes', 'yes'],
    'FCVC': [2.0, 1.0, 3.0],
    'NCP': [4.0, 1.0, 1.0],
    'CAEC': ['Frequently', 'Sometimes', 'Sometimes'],
    'SMOKE': ['no', 'no', 'no'],
    'CH2O': [2.0, 2.0, 1.0],
    'SCC': ['no', 'no', 'no'],
    'FAF': [2.0, 0.0, 0.0],
    'TUE': [1.0, 0.0, 2.0],
    'CALC': ['Sometimes', 'no', 'Sometimes'],
    'MTRANS': ['Automobile', 'Public_Transportation', 'Public_Transportation']
}

#'NObeyesdad': ['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_II']
atributos = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
entrada = pd.DataFrame(data, columns=atributos)


X_entrada = entrada

saidas = best_model.predict(X_entrada)
print(saidas)