import cv2
import os
import numpy as np
from skimage.feature import graycomatrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler

# Função para extrair características usando GLCM
def extrair_caracteristicas_glcm(imagem):
    glcm = graycomatrix(imagem, [5], [0], 256, symmetric=True, normed=True)
    return np.array([np.mean(glcm), np.std(glcm)])

# Diretório contendo as imagens
diretorio_imagens = r"C:\Users\Matheus Miquelini\Desktop\projeto\imagens"

# Inicialize listas para armazenar imagens e rótulos
imagens = []
rotulos = []

# Itere sobre as imagens no diretório
for nome_arquivo in os.listdir(diretorio_imagens):
    caminho_imagem = os.path.join(diretorio_imagens, nome_arquivo)

    # Leia a imagem em escala de cinza
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    # Extraia o rótulo do nome do arquivo ou do diretório
    rotulo = int(nome_arquivo.split("_")[0])  # Suponha que o rótulo está no início do nome do arquivo

    # Armazene a imagem e o rótulo nas listas
    imagens.append(imagem)
    rotulos.append(rotulo)

# Converta as listas em arrays numpy
imagens = np.array(imagens)
rotulos = np.array(rotulos)

# Dividir o conjunto de dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(imagens, rotulos, test_size=0.2, random_state=42)

# Aplicar subamostragem para balancear as classes
rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_treino.reshape((X_treino.shape[0], -1)), y_treino)

# Redimensionar os dados após a subamostragem
X_resampled_glcm = X_resampled.reshape((X_resampled.shape[0], *X_treino.shape[1:]))

print(f"X_resampled_glcm shape: {X_resampled_glcm.shape}")
print(f"y_resampled shape: {y_resampled.shape}")

# Inicializar e treinar o modelo SVM usando GLCM com dados subamostrados
modelo_glcm = SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)
modelo_glcm.fit(X_resampled_glcm.reshape((X_resampled_glcm.shape[0], -1)), y_resampled)

# Realizar previsões no conjunto de teste
X_teste_glcm = X_teste.reshape((X_teste.shape[0], -1))
y_pred_glcm = modelo_glcm.predict(X_teste_glcm)

# Calcular métricas para o modelo GLCM
relatorio_glcm = classification_report(y_teste, y_pred_glcm)
print("\nMétricas para o Modelo GLCM:")
print(relatorio_glcm)

# Limiar de decisão
limiar_decision = 0.8

# Ajuste das previsões para aumentar falsos negativos e diminuir falsos positivos para o modelo GLCM
y_score_glcm = modelo_glcm.decision_function(X_teste_glcm)
y_pred_glcm_adjusted = (y_score_glcm > limiar_decision).astype(int)

print("\nPontuações de Decisão para o Modelo GLCM:")
print(y_score_glcm)

# Calcular métricas para o modelo GLCM ajustado
relatorio_glcm_adjusted = classification_report(y_teste, y_pred_glcm_adjusted)
print("\nMétricas para o Modelo GLCM (Ajustado):")
print(relatorio_glcm_adjusted)
