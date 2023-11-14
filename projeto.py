import cv2
import os
import numpy as np
from skimage.feature import hog, greycomatrix
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Função para extrair características usando HOG
def extrair_caracteristicas_hog(imagem):
    fd, hog_imagem = hog(imagem, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
    hog_imagem_rescale = exposure.rescale_intensity(hog_imagem, in_range=(0, 10))
    return fd

# Função para extrair características usando GLCM
def extrair_caracteristicas_glcm(imagem):
    glcm = greycomatrix(imagem, [5], [0], 256, symmetric=True, normed=True)
    return np.array([np.mean(glcm), np.std(glcm)])

# Diretório contendo as imagens
diretorio_imagens = "caminho/do/seu/diretorio"

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

# Inicializar e treinar o modelo SVM usando HOG
X_treino_hog = np.array([extrair_caracteristicas_hog(imagem) for imagem in X_treino])
X_teste_hog = np.array([extrair_caracteristicas_hog(imagem) for imagem in X_teste])

modelo_hog = SVC(kernel='linear', C=1)
modelo_hog.fit(X_treino_hog, y_treino)
y_pred_hog = modelo_hog.predict(X_teste_hog)
precisao_hog = accuracy_score(y_teste, y_pred_hog)

# Inicializar e treinar o modelo SVM usando GLCM
X_treino_glcm = np.array([extrair_caracteristicas_glcm(imagem) for imagem in X_treino])
X_teste_glcm = np.array([extrair_caracteristicas_glcm(imagem) for imagem in X_teste])

modelo_glcm = SVC(kernel='linear', C=1)
modelo_glcm.fit(X_treino_glcm, y_treino)
y_pred_glcm = modelo_glcm.predict(X_teste_glcm)
precisao_glcm = accuracy_score(y_teste, y_pred_glcm)

# Calcular métricas para o modelo HOG
relatorio_hog = classification_report(y_teste, y_pred_hog)
print("Métricas para o Modelo HOG:")
print(relatorio_hog)

# Calcular métricas para o modelo GLCM
relatorio_glcm = classification_report(y_teste, y_pred_glcm)
print("\nMétricas para o Modelo GLCM:")
print(relatorio_glcm)

########################

# Limiar de decisão
limiar_decision = 0.8

# Ajuste das previsões para aumentar falsos negativos e diminuir falsos positivos para o modelo HOG
y_pred_hog_adjusted = (modelo_hog.predict_proba(X_teste_hog)[:, 1] > limiar_decision).astype(int)

# Ajuste das previsões para aumentar falsos negativos e diminuir falsos positivos para o modelo GLCM
y_pred_glcm_adjusted = (modelo_glcm.predict_proba(X_teste_glcm)[:, 1] > limiar_decision).astype(int)

# Calcular métricas para os modelos ajustados
relatorio_hog_adjusted = classification_report(y_teste, y_pred_hog_adjusted)
print("Métricas para o Modelo HOG (Ajustado):")
print(relatorio_hog_adjusted)

relatorio_glcm_adjusted = classification_report(y_teste, y_pred_glcm_adjusted)
print("\nMétricas para o Modelo GLCM (Ajustado):")
print(relatorio_glcm_adjusted)
