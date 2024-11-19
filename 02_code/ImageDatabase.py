from Imagen import Imagen, cv2, np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from kmeans import KMeans

class ImageDatabase:

    def __init__(self):
        self.berenjenas = []
        self.camotes = []
        self.papas = []
        self.zanahorias = []
        self.features = None
        self.labels = None
        self.reset_scaler()
        self.classifier = KMeans(n_clusters=4, max_iters=100, tol=1e-3)

    def reset_scaler(self):
        """Reset the scaler to its initial state"""
        self.scaler = StandardScaler()
        self.scaler.n_samples_seen_ = 0
        self.scaler.mean_ = None
        self.scaler.var_ = None
        self.scaler.scale_ = None

    def cargar_imagenes(self, database_path = '../03_db'):
        self.database_path = Path(database_path)
        
        for berenjena_path in self.database_path.glob('01_berenjena/*.jpg'):
            self.berenjenas.append(Imagen());
            self.berenjenas[-1].imagen_desde_archivo(berenjena_path)
        
        for camote_path in self.database_path.glob('02_camote/*.jpg'):
            self.camotes.append(Imagen())
            self.camotes[-1].imagen_desde_archivo(camote_path)
        
        for papa_path in self.database_path.glob('03_papa/*.jpg'):
            self.papas.append(Imagen())
            self.papas[-1].imagen_desde_archivo(papa_path)

        for zanahoria_path in self.database_path.glob('04_zanahoria/*.jpg'):
            self.zanahorias.append(Imagen())
            self.zanahorias[-1].imagen_desde_archivo(zanahoria_path)


    def procesar_imagenes(self):
        for berenjena in self.berenjenas:
            berenjena.procesar_imagen_individual();

        for camote in self.camotes:
            camote.procesar_imagen_individual();

        for papa in self.papas:
            papa.procesar_imagen_individual();

        for zanahoria in self.zanahorias:
            zanahoria.procesar_imagen_individual();

    
    # Funciones de debug
    def mostrar_base_de_datos(self):
        # Definir el tamaño de la ventana HD
        screen_width = 1920
        screen_height = 1080
        
        # Calcular el tamaño de cada imagen manteniendo el ratio 4:3
        num_columns = len(self.berenjenas)  # Asumimos que todas las categorías tienen la misma cantidad de imágenes
        image_width = screen_width // num_columns
        image_height = int(image_width * 4 / 3)
        if image_height * 4 > screen_height:
            image_height = screen_height // 4
            image_width = int(image_height * 3 / 4)
        
        # Redimensionar las imágenes y apilarlas
        def resize_image(image, width, height):
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        
        def draw_contours(image, contours):
            image_with_contours = image.copy()
            cv2.drawContours(image_with_contours, [contours], -1, (0, 255, 0), 10)
            return image_with_contours
        
        def stack_images(images, width, height):
            resized_images = []
            for img in images:
                if img.contours is not None:
                    image_with_contours = draw_contours(img.image, img.contours)
                else:
                    image_with_contours = img.image
                resized_image = resize_image(image_with_contours, width, height)
                resized_images.append(resized_image)
            return np.hstack(resized_images)
        
        berenjenas_row = stack_images(self.berenjenas, image_width, image_height)
        camotes_row = stack_images(self.camotes, image_width, image_height)
        papas_row = stack_images(self.papas, image_width, image_height)
        zanahorias_row = stack_images(self.zanahorias, image_width, image_height)
        
        # Apilar todas las filas verticalmente
        stacked_image = np.vstack([berenjenas_row, camotes_row, papas_row, zanahorias_row])
        
        # Mostrar la imagen apilada
        cv2.imshow('Base de Datos Completa', stacked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def guardar_caracteristicas_csv(self, filename='caracteristicas_imagenes.csv'):
        features = []
        labels = []
        feature_names = ['color_B', 'color_G', 'color_R','log_hu_moment_2']
        
        for categoria, lista in [('Berenjena', self.berenjenas), 
                               ('Camote', self.camotes),
                               ('Papa', self.papas), 
                               ('Zanahoria', self.zanahorias)]:
            for imagen in lista:
                if hasattr(imagen, 'features'):
                    features.append(imagen.features)
                    labels.append(categoria)
        
        features = np.array(features)
        # Reset and fit the scaler before transforming
        self.reset_scaler()
        normalized_features = self.scaler.fit_transform(features)
        
        df = pd.DataFrame(normalized_features, columns=feature_names)
        df['Categoria'] = labels
        # Guardar los parámetros de normalización
        np.save('scaler_params.npy', [self.scaler.mean_, self.scaler.scale_])
        df.to_csv(filename, index=False)


    def cargar_caracteristicas_csv(self, filename='caracteristicas_imagenes.csv'):
        df = pd.DataFrame(pd.read_csv(filename))
        self.features = df.iloc[:, :-1].values
        self.labels = df['Categoria'].values
        # Cargar parámetros de normalización si existen
        try:
            mean_, scale_ = np.load('scaler_params.npy')
            self.scaler.mean_ = mean_
            self.scaler.scale_ = scale_
        except:
            print("No se encontraron parámetros de normalización previos")


    def visualizar_pca(self, ax=None):
        # Aplicar PCA sobre datos normalizados
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(self.features)
        
        # Usar el axis proporcionado o crear uno nuevo si no se proporciona
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

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
            ax.scatter(
                features_3d[mask, 0],
                features_3d[mask, 1],
                features_3d[mask, 2],
                c=styles[label]['color'],
                marker=styles[label]['marker'],
                s=100,
                label=label,
                alpha=0.8,
                edgecolor='white'
            )

        # Configurar gráfico
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_zlabel('PC3', fontsize=12)
        ax.legend()
        return ax


    def entrenar_clasificador(self):
        # Inicializar y entrenar el KMeans con los datos normalizados
        self.classifier.initialize_centroids(self)
        self.classifier.fit(self.features)
        self.classifier.assign_cluster_labels(self.labels)
        return self.classifier

    def predecir_categoria(self, imagen_nueva):
        if hasattr(imagen_nueva, 'features'):
            # Normalizar las características de la nueva imagen
            features_norm = self.scaler.transform(imagen_nueva.features.reshape(1, -1))
            # Predecir usando el clasificador entrenado
            return self.classifier.predict(features_norm)
        return None



