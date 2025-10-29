import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.stats import kstest, f_oneway, kruskal
from scipy.signal import welch
from sklearn.decomposition import PCA
#preparação dos dados
def load_participant_data(participant_id,
                          data_folder=r'C:\Users\squar\PycharmProjects\ECACprodject1\FORTH_TRACE_DATASET-master'):
    participant_folder = os.path.join(data_folder, f'part{participant_id}')
    all_data = []

    # Carregar dados dos 5 dispositivos
    for device_id in range(1, 6):
        filename = f'part{participant_id}dev{device_id}.csv'
        filepath = os.path.join(participant_folder, filename)

        try:
            with open(filepath, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    try:
                        data_row = [float(value) for value in row]
                        all_data.append(data_row)
                    except ValueError:
                        continue
        except FileNotFoundError:
            print(f"Aviso: Ficheiro {filename} não encontrado.")
            continue

    data_array = np.array(all_data)
    return data_array

#---------------------------------------------
#exercicio 3.1
#---------------------------------------------
def calculate_vector_magnitude(data_array, start_col):
    x = data_array[:, start_col]
    y = data_array[:, start_col + 1]
    z = data_array[:, start_col + 2]

    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    return magnitude

def plot_boxplot_by_activity(data_array, vector_type='accel'):
    # Definir coluna inicial baseada no tipo de vetor
    if vector_type == 'accel':
        start_col = 1
        title = 'Módulo do Vetor Aceleração por Atividade'
        ylabel = 'Magnitude Aceleração'
    elif vector_type == 'gyro':
        start_col = 4
        title = 'Módulo do Vetor Giroscópio por Atividade'
        ylabel = 'Magnitude Giroscópio'
    elif vector_type == 'mag':
        start_col = 7
        title = 'Módulo do Vetor Magnetómetro por Atividade'
        ylabel = 'Magnitude Magnetómetro'
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    # Calcular o módulo do vetor
    magnitude = calculate_vector_magnitude(data_array, start_col)

    # Obter as atividades
    activities = data_array[:, 11]

    # Obter atividades únicas e ordenar
    unique_activities = np.sort(np.unique(activities))

    # Preparar dados para o boxplot
    data_by_activity = []
    for activity in unique_activities:
        mask = activities == activity
        data_by_activity.append(magnitude[mask])

    # Criar o boxplot
    plt.figure(figsize=(14, 6))
    plt.boxplot(data_by_activity, tick_labels=unique_activities.astype(int))
    plt.xlabel('Atividade')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
"""
def calcular_densidade_outliers(data_array, vector_type='accel', k=3):
    # Definir coluna inicial baseada no tipo de vetor
    if vector_type == 'accel':
        start_col = 1
    elif vector_type == 'gyro':
        start_col = 4
    elif vector_type == 'mag':
        start_col = 7
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    # Escolher o dispositivo, neste caso o pulso direito
    mask_device = data_array[:, 0] == 2
    data_filtered = data_array[mask_device]

    # Calcular o módulo do vetor
    magnitude = calculate_vector_magnitude(data_filtered, start_col)

    # Obter as atividades e ordenar as atividades únicas
    activities = data_filtered[:, 11]
    unique_activities = np.sort(np.unique(activities))

    # Calcular as densidades
    densidades = {}
    for activity in unique_activities:
        mask = activities == activity
        dados_atividade = magnitude[mask]

        outliers, _ = detectar_outliers_zscore(dados_atividade, k=k)
        no = len(outliers)
        nr = len(dados_atividade)

        if nr > 0:
            densidade = (no / nr) * 100
        else:
            densidade = 0

        densidades[int(activity)] = densidade

    return densidades
"""
#-----------------------------------------------------------------
#exercicio 3.2
#-----------------------------------------------------------------

def calcular_densidade_outliers(data_array, vector_type='accel', k=3):

    if vector_type == 'accel':
        start_col = 1
    elif vector_type == 'gyro':
        start_col = 4
    elif vector_type == 'mag':
        start_col = 7
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    # Escolher o dispositivo (pulso direito)
    mask_device = data_array[:, 0] == 2
    data_filtered = data_array[mask_device]

    magnitude = calculate_vector_magnitude(data_filtered, start_col)
    activities = data_filtered[:, 11]
    unique_activities = np.sort(np.unique(activities))

    densidades = {}
    for activity in unique_activities:
        mask = activities == activity
        dados_atividade = magnitude[mask]

        # --- Método IQR (Tukey) ---
        Q1 = np.percentile(dados_atividade, 25)
        Q3 = np.percentile(dados_atividade, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = dados_atividade[(dados_atividade < lower_bound) | (dados_atividade > upper_bound)]

        no = len(outliers)
        nr = len(dados_atividade)
        densidade = (no / nr) * 100 if nr > 0 else 0
        densidades[int(activity)] = densidade

    print(f"\nDensidades de outliers ({vector_type},):")
    for a, d in densidades.items():
        print(f"  Atividade {a}: {d:.2f}%")

    return densidades


#------------------------------------------------------------------------------------
#exercicio 3.3
#------------------------------------------------------------------------------------
def detectar_outliers_zscore(dados, k=3):
    # Converter para array NumPy
    dados = np.array(dados)

    # Calcular média e desvio padrão
    media = np.mean(dados)
    desvio = np.std(dados)

    # Evitar divisão por zero
    if desvio == 0:
        return np.array([]), np.array([])

    # Calcular Z-scores
    z_scores = (dados - media) / desvio

    # Identificar outliers
    mask_outliers = np.abs(z_scores) > k

    return dados[mask_outliers], np.where(mask_outliers)[0]

#----------------------------------------------------------------
#3.4
#----------------------------------------------------------------
def plot_outliers_zscore(data_array, vector_type='accel', k_values=[3, 3.5, 4]):
    # Definir coluna inicial baseada no tipo de vetor
    if vector_type == 'accel':
        start_col = 1
        title = 'Módulo do Vetor Aceleração por Atividade'
        ylabel = 'Magnitude Aceleração'
    elif vector_type == 'gyro':
        start_col = 4
        title = 'Módulo do Vetor Giroscópio por Atividade'
        ylabel = 'Magnitude Giroscópio'
    elif vector_type == 'mag':
        start_col = 7
        title = 'Módulo do Vetor Magnetómetro por Atividade'
        ylabel = 'Magnitude Magnetómetro'
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    # Calcular o módulo do vetor
    magnitude = calculate_vector_magnitude(data_array, start_col)

    indices = np.arange(len(magnitude))

    # Criar um gráfico por cada valor de k
    for k in k_values:
        # Deteção de outliers
        _, outlier_indices = detectar_outliers_zscore(magnitude, k=k)

        # Criar máscara de pontos não outliers
        mask_outliers = np.zeros(len(magnitude), dtype=bool)
        mask_outliers[outlier_indices] = True

        # Gráfico
        plt.figure(figsize=(12, 5))
        plt.scatter(indices[~mask_outliers], magnitude[~mask_outliers], color='blue', s=5, label='Normal')
        plt.scatter(indices[mask_outliers], magnitude[mask_outliers], color='red', s=8, label='Outliers')

        plt.title(f'{title} - Outliers (k={k})')
        plt.xlabel('Amostras')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)

#-------------------------------------------
#K-means
#-------------------------------------------

def k_means(data, n_clusters, max_iters=100, tol=1e-3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = data.shape[0]

    # Inicializar centróides aleatoriamente
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = data[indices].copy()

    for iteration in range(max_iters):
        # Calcular distâncias (vectorizado)
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Recalcular centróides
        centroids_old = centroids.copy()
        for k in range(n_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                centroids[k] = data[np.random.choice(n_samples)]

        # Verificar convergência
        if np.linalg.norm(centroids - centroids_old) < tol:
            break

    return labels, centroids

def detectar_outliers_kmeans(data_array, vector_type='accel', n_clusters=3):
    """
    3.7 - Determina outliers usando K-means.
    Pontos em clusters muito pequenos são considerados outliers.

    Parâmetros:
    -----------
    data_array : numpy.ndarray
        Array com todos os dados
    vector_type : str
        Tipo de vetor: 'accel', 'gyro' ou 'mag'
    n_clusters : int
        Número de clusters

    Retorna:
    --------
    outlier_mask : numpy.ndarray
        Máscara booleana indicando outliers
    labels : numpy.ndarray
        Etiquetas de cluster
    """

    if vector_type == 'accel':
        start_col = 1
    elif vector_type == 'gyro':
        start_col = 4
    elif vector_type == 'mag':
        start_col = 7
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    # Extrair os 3 componentes do vetor
    data_3d = data_array[:, start_col:start_col + 3]

    # Aplicar K-means
    labels, centroids = k_means(data_3d, n_clusters, random_state=42)

    # Contar pontos em cada cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Clusters com menos de 1% dos dados são outliers
    threshold = len(data_3d) * 0.03
    outlier_clusters = unique_labels[counts < threshold]

    # Criar máscara de outliers
    outlier_mask = np.isin(labels, outlier_clusters)

    n_outliers = np.sum(outlier_mask)
    density = (n_outliers / len(data_3d)) * 100

    print(f"\nK-means com {n_clusters} clusters ({vector_type}):")
    print(f"  Outliers detectados: {n_outliers}/{len(data_3d)} ({density:.2f}%)")
    print(f"  Distribuição por cluster: {dict(zip(unique_labels, counts))}")

    return outlier_mask, labels, data_3d, centroids

def plot_outliers_kmeans_3d(data_array, vector_type='accel', n_clusters_list=[3, 4, 5, 6]):
    """
    3.7 - Visualiza outliers detectados por K-means em gráficos 3D.
    Compara diferentes números de clusters.

    Parâmetros:
    -----------
    data_array : numpy.ndarray
        Array com todos os dados
    vector_type : str
        Tipo de vetor
    n_clusters_list : list
        Lista de números de clusters a testar
    """

    if vector_type == 'accel':
        title_base = 'Aceleração'
    elif vector_type == 'gyro':
        title_base = 'Giroscópio'
    elif vector_type == 'mag':
        title_base = 'Magnetómetro'
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    for n_clusters in n_clusters_list:
        outlier_mask, labels, data_3d, centroids = detectar_outliers_kmeans(
            data_array, vector_type, n_clusters
        )

        # Criar figura 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plotar pontos normais (azul)
        normal_points = data_3d[~outlier_mask]
        ax.scatter(normal_points[:, 0], normal_points[:, 1], normal_points[:, 2],
                   c='blue', s=5, alpha=0.6, label='Normal')

        # Plotar outliers (vermelho)
        outlier_points = data_3d[outlier_mask]
        ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2],
                   c='red', s=20, alpha=0.8, label='Outliers')

        # Plotar centros dos clusters (verde)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   c='green', marker='X', s=200, label='Centróides', edgecolors='black', linewidths=2)

        ax.set_xlabel('Componente X')
        ax.set_ylabel('Componente Y')
        ax.set_zlabel('Componente Z')
        ax.set_title(f'{title_base} - K-means (n_clusters={n_clusters})')
        ax.legend()

        plt.tight_layout()
        plt.show(block=False)

def comparar_kmeans_vs_zscore(data_array, vector_type='accel', n_clusters=5, k=3):
    """
    3.7 - Compara resultados do K-means com Z-score.

    Parâmetros:
    -----------
    data_array : numpy.ndarray
        Array com todos os dados
    vector_type : str
        Tipo de vetor
    n_clusters : int
        Número de clusters para K-means
    k : float
        Parâmetro k para Z-score
    """

    if vector_type == 'accel':
        start_col = 1
        label = 'Aceleração'
    elif vector_type == 'gyro':
        start_col = 4
        label = 'Giroscópio'
    elif vector_type == 'mag':
        start_col = 7
        label = 'Magnetómetro'
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    # Magnitude do vetor
    magnitude = calculate_vector_magnitude(data_array, start_col)

    # Outliers K-means
    outlier_mask_kmeans, _, _, _ = detectar_outliers_kmeans(data_array, vector_type, n_clusters)

    # Outliers Z-score
    media = np.mean(magnitude)
    desvio = np.std(magnitude)
    if desvio > 0:
        z_scores = (magnitude - media) / desvio
        outlier_mask_zscore = np.abs(z_scores) > k
    else:
        outlier_mask_zscore = np.zeros(len(magnitude), dtype=bool)

    # Estatísticas
    n_kmeans = np.sum(outlier_mask_kmeans)
    n_zscore = np.sum(outlier_mask_zscore)
    both = np.sum(outlier_mask_kmeans & outlier_mask_zscore)

    print(f"\n{'=' * 80}")
    print(f"COMPARAÇÃO: K-means vs Z-score ({label})")
    print(f"{'=' * 80}")
    print(f"Outliers K-means (n_clusters={n_clusters}): {n_kmeans} ({n_kmeans / len(magnitude) * 100:.2f}%)")
    print(f"Outliers Z-score (k={k}): {n_zscore} ({n_zscore / len(magnitude) * 100:.2f}%)")
    print(f"Outliers detectados por ambos: {both}")
    print(f"Concordância: {both / max(n_kmeans, n_zscore) * 100:.2f}% (dos detectados por K-means)")
    print(f"{'=' * 80}\n")

    return outlier_mask_kmeans, outlier_mask_zscore

# 4.1 - Testes de significância
def determinar_significancia_atividade(data_array, vector_type='accel'):
    if vector_type == 'accel':
        start_col = 1
        label = 'Aceleração'
    elif vector_type == 'gyro':
        start_col = 4
        label = 'Giroscópio'
    elif vector_type == 'mag':
        start_col = 7
        label = 'Magnetómetro'
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")

    magnitude = calculate_vector_magnitude(data_array, start_col)
    activities = data_array[:, 11]
    unique_activities = np.sort(np.unique(activities))
    grupos = [magnitude[activities == a] for a in unique_activities]

    normalidades = []
    for g in grupos:
        if len(g) > 0:
            std_g = np.std(g)
            if std_g == 0:
                normalidades.append(False)
            else:
                stat, p_val = kstest((g - np.mean(g)) / std_g, 'norm')
                normalidades.append(p_val > 0.05)
        else:
            normalidades.append(False)

    normal = all(normalidades)
    if normal:
        stat, p_value = f_oneway(*grupos)
        teste_usado = 'ANOVA (paramétrico)'
    else:
        stat, p_value = kruskal(*grupos)
        teste_usado = 'Kruskal-Wallis (não paramétrico)'

    print(f"\n{'=' * 80}")
    print(f"4.1 - Teste de significância ({label})")
    print(f"{'-' * 80}")
    print(f"Normalidade (todas as atividades normais?): {'Sim' if normal else 'Não'}")
    print(f"Teste usado: {teste_usado}")
    print(f"Estatística = {stat:.4f}, p-valor = {p_value:.6f}")
    print(f"{'=' * 80}\n")
    return {'vector': vector_type, 'teste': teste_usado, 'p_value': p_value, 'normal': normal}

# 4.2 - Features
def sliding_window_segments(data_array, window_size_sec=5, overlap=0.5, fs=50):
    window_size = int(window_size_sec * fs)
    step_size = int(window_size * (1 - overlap))
    activities = data_array[:, 11]
    segments = []
    for start in range(0, len(data_array) - window_size + 1, step_size):
        end = start + window_size
        window = data_array[start:end]
        if len(np.unique(window[:, 11])) == 1:
            segments.append(window)
    return segments

def extract_temporal_features(segment, vector_type='accel'):
    if vector_type == 'accel':
        start_col = 1
    elif vector_type == 'gyro':
        start_col = 4
    elif vector_type == 'mag':
        start_col = 7
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")
    features = []
    for i in range(start_col, start_col + 3):
        axis_data = segment[:, i]
        features.extend([np.mean(axis_data), np.std(axis_data),
                         np.max(axis_data), np.min(axis_data),
                         np.sum(axis_data ** 2) / len(axis_data)])
    return np.array(features)

def extract_spectral_features(segment, vector_type='accel', fs=50):
    if vector_type == 'accel':
        start_col = 1
    elif vector_type == 'gyro':
        start_col = 4
    elif vector_type == 'mag':
        start_col = 7
    else:
        raise ValueError("vector_type deve ser 'accel', 'gyro' ou 'mag'")
    features = []
    for i in range(start_col, start_col + 3):
        axis_data = segment[:, i]
        freqs, psd = welch(axis_data, fs=fs, nperseg=min(256, len(axis_data)))
        features.extend([np.mean(psd), freqs[np.argmax(psd)]])
    return np.array(features)

def extract_feature_vector(segment, vector_type='accel', fs=50):
    return np.concatenate([extract_temporal_features(segment, vector_type),
                           extract_spectral_features(segment, vector_type, fs)])

def extract_feature_set(data_array, vector_type='accel', window_size_sec=5, overlap=0.5, fs=50):
    segments = sliding_window_segments(data_array, window_size_sec, overlap, fs)
    X, y = [], []
    for segment in segments:
        X.append(extract_feature_vector(segment, vector_type, fs))
        y.append(int(segment[0, 11]))
    return np.array(X), np.array(y)

# 4.3 - PCA
def aplicar_pca(feature_set, n_components=0.95, vector_type='accel'):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(feature_set)

    print(f"\n{'=' * 80}")
    print(f"4.3 - PCA ({vector_type})")
    print(f"{'-' * 80}")
    print(f"Número de features originais: {feature_set.shape[1]}")
    print(f"Número de componentes principais: {X_pca.shape[1]}")
    print(f"Variância explicada por cada componente: {pca.explained_variance_ratio_}")
    print(f"Variância acumulada: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"{'=' * 80}\n")

    return X_pca, pca
# 4.5 - Fisher Score e ReliefF implementados do zero
def fisher_score(X, y):
    """
    Calcula o Fisher Score para cada feature.
    X : ndarray (amostras x features)
    y : ndarray (labels)
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    scores = np.zeros(n_features)

    for j in range(n_features):
        feature = X[:, j]
        overall_mean = np.mean(feature)
        num, den = 0, 0
        for c in classes:
            mask = y == c
            class_data = feature[mask]
            if len(class_data) > 1:
                class_mean = np.mean(class_data)
                class_var = np.var(class_data)
                num += len(class_data) * (class_mean - overall_mean) ** 2
                den += len(class_data) * class_var
        scores[j] = num / den if den > 0 else 0

    return scores


def reliefF(X, y, n_neighbors=10):
    """
    Implementação simplificada do algoritmo ReliefF.
    Usa distância euclidiana entre amostras.
    """
    from sklearn.neighbors import NearestNeighbors

    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)
    neighbors = nn.kneighbors(X, return_distance=False)

    for i in range(n_samples):
        label = y[i]
        hits = X[neighbors[i][1:]][y[neighbors[i][1:]] == label]
        misses = X[neighbors[i][1:]][y[neighbors[i][1:]] != label]

        if len(hits) > 0:
            diff_hit = np.abs(X[i] - np.mean(hits, axis=0))
        else:
            diff_hit = np.zeros(n_features)

        if len(misses) > 0:
            diff_miss = np.abs(X[i] - np.mean(misses, axis=0))
        else:
            diff_miss = np.zeros(n_features)

        scores += diff_miss - diff_hit

    scores = np.maximum(scores, 0)
    return scores


def selecionar_melhores_features(X, y, metodo='fisher', top_n=10):
    """
    4.6 - Identifica as melhores features segundo Fisher Score ou ReliefF.
    """
    if metodo.lower() == 'fisher':
        scores = fisher_score(X, y)
        nome_metodo = "Fisher Score"
    elif metodo.lower() == 'relieff':
        scores = reliefF(X, y)
        nome_metodo = "ReliefF"
    else:
        raise ValueError("Método deve ser 'fisher' ou 'relieff'.")

    top_idx = np.argsort(scores)[::-1][:top_n]

    print(f"\n{'=' * 80}")
    print(f"4.6 - Top {top_n} Features segundo {nome_metodo}")
    print(f"{'-' * 80}")
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank:2d}. Feature {idx:3d}  |  Score = {scores[idx]:.5f}")
    print(f"{'=' * 80}\n")

    return top_idx, scores

import matplotlib.pyplot as plt
import gc
from main import *

def run_option(func, *args, **kwargs):
    """Executa cada tarefa de forma isolada, como na main original."""
    plt.close('all')      # fecha gráficos antigos
    gc.collect()          # limpa memória antiga
    func(*args, **kwargs) # executa a função
    plt.show()            # mostra e bloqueia até o utilizador fechar
    plt.close('all')      # fecha o gráfico antes de voltar ao menu
    gc.collect()          # limpa novamente

def main():
    participant_id = int(input("Introduz o ID do participante (1–15): "))
    data = load_participant_data(participant_id)
    print(f"✅ Dados carregados: {data.shape}")

    # Variáveis globais de features e PCA (para as opções 7–9)
    X_accel = X_gyro = X_mag = y_accel = y_gyro = y_mag = None

    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1  - 3.1 Boxplots das variáveis transformadas")
        print("2  - 3.2 Densidade de outliers (IQR, pulso direito)")
        print("3  - 3.4 Deteção e plot de outliers (Z-score)")
        print("4  - 3.6 Execução do algoritmo K-means (sem plot)")
        print("5  - 3.7 Outliers com K-means")
        print("6  - Comparar K-means vs Z-score")
        print("7  - 4.1 Teste de significância estatística")
        print("8  - 4.2 Extração de features")
        print("9  - 4.3 Aplicar PCA")
        print("10  - 4.5–4.6 Seleção de features")
        print("11 - 🔁 Executar tudo (3.1 a 4.6)")
        print("0  - ❌ Sair")

        opcao = input("\nEscolhe uma opção: ")

        # === 3.1 ===
        if opcao == "1":
            for v in ['accel', 'gyro', 'mag']:
                run_option(plot_boxplot_by_activity, data, vector_type=v)

        # === 3.2 ===
        elif opcao == "2":
            for v in ['accel', 'gyro', 'mag']:
                run_option(calcular_densidade_outliers, data, vector_type=v, k=3)

        # === 3.4 ===
        elif opcao == "3":
            for v in ['accel', 'gyro', 'mag']:
                run_option(plot_outliers_zscore, data, vector_type=v, k_values=[3, 3.5, 4])

        # === 3.6 ===
        elif opcao == "4":
            for v in ['accel', 'gyro', 'mag']:
                print(f"\n⚙️  A executar K-means para {v}...")
                data_3d = data[:, {'accel': 1, 'gyro': 4, 'mag': 7}[v]:{'accel': 1, 'gyro': 4, 'mag': 7}[v] + 3]
                for n_clusters in [2, 3, 4, 5, 6, 7, 8]:
                    _, centroids = k_means(data_3d, n_clusters, random_state=42)
                    print(f" - K={n_clusters}: {len(centroids)} centróides calculados.")
            print(" Execução do 3.6 concluída!")

        # === 3.7 ===
        elif opcao == "5":
            for v in ['accel', 'gyro', 'mag']:
                run_option(plot_outliers_kmeans_3d, data, vector_type=v, n_clusters_list=[3, 5, 6])

        # === Comparar K-means vs Z-score ===
        elif opcao == "6":
            for v in ['accel', 'gyro', 'mag']:
                run_option(comparar_kmeans_vs_zscore, data, vector_type=v, n_clusters=5, k=3)

        # === 4.1 ===
        elif opcao == "7":
            for v in ['accel', 'gyro', 'mag']:
                run_option(determinar_significancia_atividade, data, vector_type=v)

        # === 4.2 ===
        elif opcao == "8":
            print("\n📦 A extrair features (pode demorar)...")
            X_accel, y_accel = extract_feature_set(data, vector_type='accel')
            X_gyro, y_gyro = extract_feature_set(data, vector_type='gyro')
            X_mag, y_mag = extract_feature_set(data, vector_type='mag')
            print(f"✅ Features aceleração: {X_accel.shape}")
            print(f"✅ Features giroscópio: {X_gyro.shape}")
            print(f"✅ Features magnetómetro: {X_mag.shape}")

        # === 4.3 ===
        elif opcao == "9":
            if X_accel is None:
                print(" Executa primeiro a opção 7 (extração de features).")
                continue
            run_option(aplicar_pca, X_accel, n_components=0.95, vector_type='accel')
            run_option(aplicar_pca, X_gyro, n_components=0.95, vector_type='gyro')
            run_option(aplicar_pca, X_mag, n_components=0.95, vector_type='mag')

        # === 4.5–4.6 ===
        elif opcao == "10":
            if X_accel is None:
                print("️ Executa primeiro a opção 7 (extração de features).")
                continue
            run_option(selecionar_melhores_features, X_accel, y_accel, metodo='fisher', top_n=10)
            run_option(selecionar_melhores_features, X_accel, y_accel, metodo='relieff', top_n=10)

        # === Executar tudo ===
        elif opcao == "11":
            print("\n A executar todas as etapas (3.1 → 4.6)...")
            etapas = [
                (plot_boxplot_by_activity, dict(vector_type='accel')),
                (plot_boxplot_by_activity, dict(vector_type='gyro')),
                (plot_boxplot_by_activity, dict(vector_type='mag')),
                (calcular_densidade_outliers, dict(vector_type='accel', k=3)),
                (calcular_densidade_outliers, dict(vector_type='gyro', k=3)),
                (calcular_densidade_outliers, dict(vector_type='mag', k=3)),
                (plot_outliers_zscore, dict(vector_type='accel', k_values=[3, 3.5, 4])),
                (plot_outliers_zscore, dict(vector_type='gyro', k_values=[3, 3.5, 4])),
                (plot_outliers_zscore, dict(vector_type='mag', k_values=[3, 3.5, 4])),
                (plot_outliers_kmeans_3d, dict(vector_type='accel', n_clusters_list=[3, 5, 6])),
                (plot_outliers_kmeans_3d, dict(vector_type='gyro', n_clusters_list=[3, 5, 6])),
                (plot_outliers_kmeans_3d, dict(vector_type='mag', n_clusters_list=[3, 5, 6])),
                (determinar_significancia_atividade, dict(vector_type='accel')),
                (determinar_significancia_atividade, dict(vector_type='gyro')),
                (determinar_significancia_atividade, dict(vector_type='mag')),
            ]
            for func, kwargs in etapas:
                run_option(func, data, **kwargs)

            X_accel, y_accel = extract_feature_set(data, vector_type='accel')
            X_gyro, y_gyro = extract_feature_set(data, vector_type='gyro')
            X_mag, y_mag = extract_feature_set(data, vector_type='mag')

            run_option(aplicar_pca, X_accel, n_components=0.95, vector_type='accel')
            run_option(aplicar_pca, X_gyro, n_components=0.95, vector_type='gyro')
            run_option(aplicar_pca, X_mag, n_components=0.95, vector_type='mag')

            run_option(selecionar_melhores_features, X_accel, y_accel, metodo='fisher', top_n=10)
            run_option(selecionar_melhores_features, X_accel, y_accel, metodo='relieff', top_n=10)
            print(" Execução completa!")

        elif opcao == "0":
            print("\n A terminar o programa...")
            break
        else:
            print(" Opção inválida. Tenta novamente.")

if __name__ == "__main__":
    main()
