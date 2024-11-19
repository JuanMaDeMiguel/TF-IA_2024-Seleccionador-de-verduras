from ImageDatabase import ImageDatabase
from AudioDatabase import AudioDatabase
from pathlib import Path
import os
from tkinter import filedialog, Tk
import cv2
import numpy as np
from Imagen import Imagen
from Audio import Audio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

class Vista:
    def __init__(self):
        self.image_database = ImageDatabase()
        self.audio_database = AudioDatabase()
        self.cargar_bases_datos()

    def cargar_bases_datos(self):
        # Cargar datos desde CSV
        self.image_database.cargar_caracteristicas_csv('caracteristicas_imagenes.csv')
        self.image_database.entrenar_clasificador()
        
        self.audio_database.cargar_caracteristicas_csv('caracteristicas_audios.csv')
        self.audio_database.entrenar_clasificador()

    def mostrar_menu(self):
        while True:
            print("\n=== MENÚ PRINCIPAL ===")
            print("1. Volver a analizar base de datos de imágenes")
            print("2. Volver a analizar base de datos de audios")
            print("3. Mostrar segregación de imágenes")
            print("4. Mostrar segregación de audios")
            print("5. Medir eficiencia de reconocimiento de imágenes")
            print("6. Medir eficiencia de reconocimiento de audios")
            print("7. Ejecutar seleccionador de verduras")
            print("8. Salir")
            
            opcion = input("\nSeleccione una opción: ")
            
            if opcion == "1":
                self.analizar_imagenes()
            elif opcion == "2":
                self.analizar_audios()
            elif opcion == "5":
                self.medir_eficiencia_imagenes()
            elif opcion == "6":
                self.medir_eficiencia_audios()
            elif opcion == "7":
                self.ejecutar_seleccionador()
            elif opcion == "8":
                break
            elif opcion == "3":
                self.mostrar_segregacion_imagenes()
            elif opcion == "4":
                self.audio_database.visualizar_pca()
            else:
                print("Opción no válida")

    def analizar_imagenes(self):
        self.image_database.cargar_imagenes()
        self.image_database.procesar_imagenes()
        self.image_database.guardar_caracteristicas_csv('caracteristicas_imagenes.csv')
        self.image_database.mostrar_base_de_datos()
        self.image_database.cargar_caracteristicas_csv('caracteristicas_imagenes.csv')
        self.image_database.entrenar_clasificador()

    def analizar_audios(self):
        self.audio_database.cargar_audios()
        self.audio_database.procesar_audios()
        self.audio_database.guardar_caracteristicas_csv('caracteristicas_audios.csv')
        self.audio_database.cargar_caracteristicas_csv('caracteristicas_audios.csv')
        self.audio_database.entrenar_clasificador()
    
    def mostrar_segregacion_imagenes(self):
        # Create a single figure with two subplots side by side
        fig = plt.figure(figsize=(20, 8))
        
        # Left subplot for raw data PCA
        ax1 = fig.add_subplot(121, projection='3d')
        self.image_database.visualizar_pca(ax1)
        ax1.set_title('PCA de características originales')
        
        # Right subplot for kmeans clustering
        ax2 = fig.add_subplot(122, projection='3d')
        self.image_database.classifier.visualizar_pca(ax2)
        ax2.set_title('PCA de clusters K-means')
        
        plt.tight_layout()
        plt.show()

    def medir_eficiencia_imagenes(self):
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        directorio_actual = Path('../07_db_eficiencia/fotos')
        print("\n=== MENÚ DE EFICIENCIA DE IMÁGENES ===")
        print("1. Base de datos")
        print("2. Set de prueba")
        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            directorio_actual = Path('../07_db_eficiencia/fotos/db')
        elif opcion == "2":
            directorio_actual = Path('../07_db_eficiencia/fotos/set_prueba')
        else:
            print("Opción no válida")
            return
        total_imagenes = 0
        aciertos = 0
        errores = []

        for archivo in os.listdir(directorio_actual):
            if archivo.endswith('.jpg') or archivo.endswith('.jpeg') or archivo.endswith('.png'):
                total_imagenes += 1
                # Determinar la etiqueta real basada en el nombre del archivo
                etiqueta_real = ''
                if 'b' in archivo.lower():
                    etiqueta_real = 'Berenjena'
                elif 'c' in archivo.lower():
                    etiqueta_real = 'Camote'
                elif 'z' in archivo.lower():
                    etiqueta_real = 'Zanahoria'
                elif 'p' in archivo.lower():
                    etiqueta_real = 'Papa'

                # Procesar la imagen y obtener predicción
                imagen_test = Imagen()
                imagen_test.imagen_desde_archivo(os.path.join(directorio_actual, archivo))
                imagen_test.procesar_imagen_individual()
                
                features_norm = self.image_database.scaler.transform(imagen_test.features.reshape(1, -1))
                cluster_index = self.image_database.classifier.predict(features_norm)
                prediccion = self.image_database.classifier.cluster_labels[cluster_index]

                if prediccion == etiqueta_real:
                    aciertos += 1
                else:
                    errores.append(archivo)

        eficiencia = (aciertos / total_imagenes) * 100 if total_imagenes > 0 else 0
        print(f"\nEficiencia del clasificador de imágenes: {eficiencia:.2f}%")
        print(f"Total de imágenes analizadas: {total_imagenes}")
        print(f"Total de aciertos: {aciertos}")

        if errores:
            print("\nArchivos con predicción incorrecta:")
            for archivo in errores:
                print(f"- {archivo}")


    def medir_eficiencia_audios(self):
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        directorio_actual = Path('../07_db_eficiencia/audio')
        print("\n=== MENÚ DE EFICIENCIA DE AUDIOS ===")
        print("1. Base de datos")
        print("2. Set de prueba")
        print("3. Audios autor")
        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            directorio_actual = Path('../07_db_eficiencia/audio/db')
        elif opcion == "2":
            directorio_actual = Path('../07_db_eficiencia/audio/set_prueba')
        elif opcion == "3":
            directorio_actual = Path('../07_db_eficiencia/audio/mios')
        else:
            print("Opción no válida")
            return
        total_audios = 0
        aciertos = 0
        errores = []

        for archivo in os.listdir(directorio_actual):
            if archivo.endswith('.wav'):
                total_audios += 1
                # Determinar la etiqueta real basada en el nombre del archivo
                etiqueta_real = ''
                if 'b' in archivo.lower():
                    etiqueta_real = 'Berenjena'
                elif 'c' in archivo.lower():
                    etiqueta_real = 'Camote'
                elif 'p' in archivo.lower():
                    etiqueta_real = 'Papa'
                elif 'z' in archivo.lower():
                    etiqueta_real = 'Zanahoria'

                # Procesar el audio y obtener predicción
                audio_test = Audio()
                audio_test.audio_desde_archivo(os.path.join(directorio_actual, archivo))
                audio_test.analisis_completo(audio_test.audio)
                # audio_test.mostrar_pasos_analisis_audio()
                
                features_audio = np.array(audio_test.features).reshape(1, -1)
                features_audio_norm = self.audio_database.scaler.transform(features_audio)
                features_audio_reducidas = features_audio_norm[:, self.audio_database.best_feature_indices]
                prediccion = self.audio_database.classifier.predict(features_audio_reducidas)[0]

                if prediccion == etiqueta_real:
                    aciertos += 1
                else:
                    errores.append(archivo)

        eficiencia = (aciertos / total_audios) * 100 if total_audios > 0 else 0
        print(f"\nEficiencia del clasificador de audio: {eficiencia:.2f}%")
        print(f"Total de audios analizados: {total_audios}")
        print(f"Total de aciertos: {aciertos}")
        
        if errores:
            print("\nArchivos con predicción incorrecta:")
            for archivo in errores:
                print(f"- {archivo}")

    def ejecutar_seleccionador(self):
        imagenes_cargadas = []
        features_normalizadas = []

        # Abrir un único diálogo para seleccionar las 4 imágenes
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        initial_dir = os.path.abspath('../06_prueba')
        imagen_paths = filedialog.askopenfilenames(
            title="Seleccionar las 4 imágenes (usar Ctrl+Click)",
            initialdir=initial_dir,
            filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
        )
        root.destroy()

        if len(imagen_paths) != 4:
            print("Error: Debe seleccionar exactamente 4 imágenes")
            return

        # Procesar las 4 imágenes seleccionadas
        for imagen_path in imagen_paths:
            imagen_a_asignar = Imagen()
            imagen_a_asignar.imagen_desde_archivo(imagen_path)
            imagen_a_asignar.procesar_imagen_individual()
            imagen_a_asignar.mostrar_proceso()
            
            features_norm = self.image_database.scaler.transform(imagen_a_asignar.features.reshape(1, -1))
            imagenes_cargadas.append(imagen_a_asignar.resized_image)
            features_normalizadas.append(features_norm)

        # Calcular matriz de distancias entre cada imagen y cada centroide
        features_stack = np.vstack(features_normalizadas)
        distancias = np.zeros((4, 4))  # 4 imágenes x 4 centroides
        for i in range(4):
            for j in range(4):
                distancias[i, j] = np.sqrt(np.sum((features_stack[i] - self.image_database.classifier.centroids[j])**2))

        # Reemplazar la sección de asignación greedy por el algoritmo húngaro
        img_indices, cluster_indices = linear_sum_assignment(distancias)
        
        # Crear las asignaciones basadas en el resultado del algoritmo húngaro
        asignaciones = [-1] * 4
        for img_idx, cluster_idx in zip(img_indices, cluster_indices):
            asignaciones[img_idx] = cluster_idx

        # Mostrar resultados
        etiquetas_imagenes = []
        for i in range(4):
            cluster_label = self.image_database.classifier.cluster_labels[asignaciones[i]]
            etiquetas_imagenes.append(cluster_label)
            print(f"Imagen {i+1} clasificada como: {cluster_label}")

        # Crear figura compuesta
        fig = plt.figure(figsize=(15, 8))
        
        # Subplot principal para el PCA
        ax_pca = fig.add_subplot(121, projection='3d')
        self.image_database.classifier.visualizar_pca(ax_pca)
        
        # Añadir los nuevos puntos al gráfico PCA
        pca = PCA(n_components=3)
        pca.fit(self.image_database.classifier.centroids)
        transformed_points = pca.transform(features_stack)
        
        for i in range(4):
            ax_pca.scatter(transformed_points[i, 0], 
                          transformed_points[i, 1], 
                          transformed_points[i, 2],
                          marker='x', color='red', s=200)
        
        # Añadir las imágenes en miniatura
        for i in range(4):
            ax_img = fig.add_subplot(4, 4, 4*(i+1))
            ax_img.imshow(cv2.cvtColor(imagenes_cargadas[i], cv2.COLOR_BGR2RGB))
            ax_img.axis('off')
            ax_img.set_title(f'{etiquetas_imagenes[i]}', fontsize=8)
        
        plt.tight_layout()
        plt.show()

        # Selección de audio (continúa igual)
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        audio_path = filedialog.askopenfilename(
            title="Seleccionar audio",
            initialdir=initial_dir,
            filetypes=(("Archivos de audio", "*.wav"), ("Todos los archivos", "*.*"))
        )
        root.destroy()

        if audio_path:
            self.procesar_audio_seleccionado(audio_path, imagenes_cargadas, etiquetas_imagenes)

    def procesar_audio_seleccionado(self, audio_path, imagenes_cargadas, etiquetas_imagenes):
        audio_a_predecir = Audio()
        audio_a_predecir.audio_desde_archivo(audio_path)
        audio_a_predecir.analisis_completo(audio_a_predecir.audio)
        
        if hasattr(audio_a_predecir, 'features'):
            features_audio = np.array(audio_a_predecir.features)
            features_audio_norm = self.audio_database.scaler.transform(features_audio.reshape(1, -1))
            features_audio_norm = features_audio_norm.reshape(-1, 1)
            features_audio_reducidas = features_audio_norm[self.audio_database.best_feature_indices]
            prediccion_audio = self.audio_database.classifier.predict(features_audio_reducidas)[0]
            
            print(f"El audio pertenece a la categoría: {prediccion_audio}")
            
            if prediccion_audio in etiquetas_imagenes:
                indices_imagenes = [idx for idx, label in enumerate(etiquetas_imagenes) if label == prediccion_audio]
                for idx in indices_imagenes:
                    imagen_correspondiente = imagenes_cargadas[idx]
                    cv2.imshow(f"Imagen correspondiente a {prediccion_audio}", imagen_correspondiente)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("No hay imágenes cargadas que correspondan a la categoría predicha.")
