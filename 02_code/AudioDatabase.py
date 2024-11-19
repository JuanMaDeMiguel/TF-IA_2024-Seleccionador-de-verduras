from Audio import Audio
from knn import KNN
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from itertools import combinations
from multiprocessing import Pool, cpu_count


class AudioDatabase:
    def __init__(self):
        self.berenjenas = []
        self.camotes = []
        self.papas = []
        self.zanahorias = []
        self.best_feature_indices = None 
        self.classifier = KNN(n_neighbors=10)
        self._init_scaler()

    def _init_scaler(self):
        """Initialize scaler with required attributes"""
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        # Initialize required attributes for the scaler
        self.scaler.n_samples_seen_ = 0
        self.scaler.mean_ = None
        self.scaler.var_ = None
        self.scaler.scale_ = None

    def cargar_audios(self, database_path='../04_db_audio'):
        self.database_path = Path(database_path)
        self.berenjenas = []
        self.camotes = []
        self.papas = []
        self.zanahorias = []
        self.best_feature_indices = None 
        self.classifier = KNN(n_neighbors=10)
        self._init_scaler()

        for berenjena_path in self.database_path.glob('01_berenjena/*.wav'):
            self.berenjenas.append(Audio())
            self.berenjenas[-1].audio_desde_archivo(berenjena_path)
        
        for camote_path in self.database_path.glob('02_camote/*.wav'):
            self.camotes.append(Audio())
            self.camotes[-1].audio_desde_archivo(camote_path)
        
        for papa_path in self.database_path.glob('03_papa/*.wav'):
            self.papas.append(Audio())
            self.papas[-1].audio_desde_archivo(papa_path)

        for zanahoria_path in self.database_path.glob('04_zanahoria/*.wav'):
            self.zanahorias.append(Audio())
            self.zanahorias[-1].audio_desde_archivo(zanahoria_path)

    def procesar_audios(self):
        for berenjena in self.berenjenas:
            berenjena.analisis_completo(berenjena.audio)
        
        for camote in self.camotes:
            camote.analisis_completo(camote.audio)
        
        for papa in self.papas:
            papa.analisis_completo(papa.audio)
        
        for zanahoria in self.zanahorias:
            zanahoria.analisis_completo(zanahoria.audio)
            
    

    def evaluar_segregacion(self, features, labels):
        """Evalúa la calidad de la segregación"""
        try:
            silhouette = silhouette_score(features, labels)
            calinski = calinski_harabasz_score(features, labels)
            return silhouette, calinski
        except:
            return -1, -1


    def evaluar_combinacion(self, args):
        """Helper function for parallel processing"""
        indices, features, labels = args
        silhouette, _ = self.evaluar_segregacion(features[:, list(indices)], labels)
        return (indices, silhouette)
        
        


    def obtener_mejores_caracteristicas(self):
        # Recolectar y preparar datos
        features = []
        labels = []
        for categoria, lista in [('Berenjena', self.berenjenas), 
                               ('Camote', self.camotes),
                               ('Papa', self.papas), 
                               ('Zanahoria', self.zanahorias)]:
            for audio in lista:
                features_array = np.array(audio.features)
                if len(features_array.shape) > 1:
                    flattened_features = features_array.reshape(features_array.shape[0], -1)[0]
                else:
                    flattened_features = features_array
                features.append(flattened_features)
                labels.append(categoria)
        
        features = np.array(features)
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
            
        # Reset and initialize scaler before fitting
        self._init_scaler()
        features = self.scaler.fit_transform(features)
        
        # Pre-evaluación de características individuales
        scores = []
        for i in range(features.shape[1]):
            feature = features[:, i:i+1]
            silhouette, calinski = self.evaluar_segregacion(feature, labels)
            scores.append((i, silhouette, calinski))
        
        # Ordenar y seleccionar top 30
        scores.sort(key=lambda x: x[1], reverse=True)
        top_30_indices = [idx for idx, _, _ in scores[:15]]
        
        # Preparar combinaciones para procesamiento paralelo
        all_combinations = []
        for r in range(1, 6):
            all_combinations.extend(combinations(top_30_indices, r))

        batch_size = 1000
        total_combinations = len(all_combinations)
        
        # Inicializar el pool de procesos
        n_processors = cpu_count()
        pool = Pool(processes=n_processors)
        
        best_combination = None
        best_score = -1
        early_stop_threshold = 0.95  # Umbral para early stopping
        
        print(f"Evaluando {total_combinations} combinaciones usando {n_processors} procesos...")
        
        # Procesar por lotes
        for batch_start in range(0, total_combinations, batch_size):
            if best_score > early_stop_threshold:
                print(f"Found excellent score ({best_score:.3f}). Early stopping...")
                break
                
            batch_end = min(batch_start + batch_size, total_combinations)
            batch_combinations = all_combinations[batch_start:batch_end]
            
            # Preparar argumentos para el procesamiento paralelo
            args = [(comb, features, labels) for comb in batch_combinations]
            
            # Procesar lote en paralelo
            results = pool.map(self.evaluar_combinacion, args)
            
            # Actualizar mejor combinación
            for indices, score in results:
                if score > best_score:
                    best_score = score
                    best_combination = indices
            
            if batch_start % 10 == 0:
                print(f"Progreso: {batch_start}/{total_combinations} combinaciones evaluadas")
                print(f"Mejor score hasta ahora: {best_score:.3f}")
        
        pool.close()
        pool.join()
        
        print(f"\nMejor combinación encontrada. Score: {best_score:.3f}")
        
        # Crear nombres descriptivos
        feature_names = []
        for idx in best_combination:
            segment_num = idx // 101
            feature_num = idx % 101
            feature_names.append(f"Seg{segment_num+1}_Feat{feature_num+1}")
        
        # Extraer mejores características
        best_features = features[:, list(best_combination)]
        
        return best_features, feature_names, labels, best_combination  # Añadimos best_combination

    def guardar_caracteristicas_csv(self, filename='caracteristicas_audio.csv'):
        features, feature_names, labels, best_indices = self.obtener_mejores_caracteristicas()
        self.best_feature_indices = best_indices
        
        # Ya no necesitamos normalizar aquí porque ya están normalizadas
        df = pd.DataFrame(features, columns=feature_names)
        df['Categoria'] = labels
        df.to_csv(filename, index=False)
        
        # Guardar los índices y parámetros de normalización
        np.save('best_audio_feature_indices.npy', self.best_feature_indices)
        np.save('audio_scaler_params.npy', [self.scaler.mean_, self.scaler.scale_])

    def cargar_caracteristicas_csv(self, filename='caracteristicas_audio.csv'):
        df = pd.DataFrame(pd.read_csv(filename))
        self.features = df.iloc[:, :-1].values
        self.labels = df['Categoria'].values
        
        # Ensure features have correct shape
        if len(self.features.shape) == 1:
            self.features = self.features.reshape(-1, 1)
            
        # Load indices and normalization parameters
        try:
            self.best_feature_indices = np.load('best_audio_feature_indices.npy')
            mean_, scale_ = np.load('audio_scaler_params.npy')
            self.scaler.mean_ = mean_
            self.scaler.scale_ = scale_
        except FileNotFoundError:
            print("Warning: Could not load normalization parameters")

    def visualizar_pca(self):
        # Determinar el número de componentes para PCA
        n_components = min(3, self.features.shape[1])
        
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(self.features)
        
        # Crear gráfico
        fig = plt.figure(figsize=(12, 8))
        
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        
        # Estilos para cada categoría
        styles = {
            'Berenjena': {'color': 'black', 'marker': 'o'},
            'Camote': {'color': 'violet', 'marker': 'o'},
            'Papa': {'color': 'yellow', 'marker': 'o'},
            'Zanahoria': {'color': 'orange', 'marker': 'o'}
        }

        # Plotear cada categoría
        for label in styles.keys():
            mask = np.array(self.labels) == label
            if n_components == 3:
                ax.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    features_pca[mask, 2],
                    c=styles[label]['color'],
                    marker=styles[label]['marker'],
                    s=100,
                    label=label,
                    alpha=0.8,
                    edgecolor='white'
                )
            elif n_components == 2:
                ax.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    c=styles[label]['color'],
                    marker=styles[label]['marker'],
                    s=100,
                    label=label,
                    alpha=0.8,
                    edgecolor='white'
                )
            else:
                ax.scatter(
                    features_pca[mask, 0],
                    np.zeros_like(features_pca[mask, 0]),
                    c=styles[label]['color'],
                    marker=styles[label]['marker'],
                    s=100,
                    label=label,
                    alpha=0.8,
                    edgecolor='white'
                )

        # Configurar gráfico
        ax.set_xlabel('PC1', fontsize=12)
        if n_components > 1:
            ax.set_ylabel('PC2', fontsize=12)
        if n_components == 3:
            ax.set_zlabel('PC3', fontsize=12)
        ax.legend(bbox_to_anchor=(1.15, 1))
        plt.title('PCA de la mejor combinación de características', pad=20)
        
        plt.tight_layout()
        plt.show()

    def entrenar_clasificador(self):
        self.classifier.fit(self.features, self.labels)

    def predecir_categoria(self, audio_nuevo):
        if hasattr(audio_nuevo, 'features'):
            # Obtener características y aplanarlas
            features_audio = np.array(audio_nuevo.features).flatten()
            # Seleccionar las mejores características
            features_reducidas = features_audio[self.best_feature_indices]
            # Normalizar usando los mismos parámetros del entrenamiento
            features_norm = self.scaler.transform(features_reducidas.reshape(1, -1))
            # Predecir usando el clasificador
            return self.classifier.predict(features_norm)[0]
        return None
