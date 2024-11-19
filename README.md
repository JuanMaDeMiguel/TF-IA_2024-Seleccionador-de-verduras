# Proyecto de Procesamiento de Imágenes y Audio

Este proyecto implementa un sistema para el procesamiento y clasificación de **imágenes** y **audios**, utilizando técnicas de **aprendizaje automático** y **procesamiento de señales**. Se enfoca en clasificar diferentes categorías de vegetales: **berenjenas**, **camotes**, **papas** y **zanahorias**, a partir de imágenes y grabaciones de audio.

---

## Tabla de Contenidos

- Descripción General
- Características Principales
- Estructura del Proyecto
- Requisitos
- Uso
- Funciones Implementadas
- Contribuciones

---

## Descripción General

El objetivo del proyecto es desarrollar un sistema capaz de:

- Procesar y clasificar **imágenes** de vegetales utilizando algoritmos de agrupamiento (*clustering*).
- Procesar y clasificar **audios** asociados a los vegetales utilizando algoritmos de clasificación supervisada.
- Proporcionar una interfaz para **medir la eficiencia** de los clasificadores y visualizar los resultados.
- Implementar un **seleccionador** que asocia imágenes con audios correspondientes.

---

## Características Principales

### Procesamiento de Imágenes

- **Segmentación y extracción de características** mediante **K-Means**.
- **Reducción de dimensiones** usando **PCA** para visualización.
- **Clasificación no supervisada** de imágenes.

### Procesamiento de Audios

- Lectura y preprocesamiento de archivos **WAV**.
- Extracción de características como **MFCC**, contraste espectral, ZCR y duración.
- **Clasificación supervisada** utilizando **K-Nearest Neighbors (K-NN)**.

### Interfaz de Usuario

- Menú interactivo en consola para seleccionar opciones.
- Visualización de resultados y eficiencia de los clasificadores.
- Seleccionador que relaciona imágenes con audios basados en las predicciones.

---

## Estructura del Proyecto

```plaintext
├── 01_UML                     # Diagramas UML del proyecto
├── 02_code                    # Código fuente del proyecto
│   ├── Audio.py               # Clase para procesar audios
│   ├── AudioDatabase.py       # Gestión de la base de datos de audios
│   ├── ImageDatabase.py       # Gestión de la base de datos de imágenes
│   ├── Imagen.py              # Clase para procesar individualmente cada imagen
│   ├── kmeans.py              # Implementación del algoritmo K-Means
│   ├── knn.py                 # Implementación del algoritmo K-Nearest Neighbors
│   ├── main.py                # Punto de entrada principal del programa
│   └── vista.py               # Gestiona la interacción con el usuario
├── 03_db                      # Base de datos de imágenes
│   ├── 01_berenjena           # Imágenes de berenjenas
│   ├── 02_camote              # Imágenes de camotes
│   ├── 03_papa                # Imágenes de papas
│   └── 04_zanahoria           # Imágenes de zanahorias
├── 04_db_audio                # Base de datos de audios
│   ├── 01_berenjena           # Audios de berenjenas
│   ├── 02_camote              # Audios de camotes
│   ├── 03_papa                # Audios de papas
│   └── 04_zanahoria           # Audios de zanahorias
├── 06_prueba                  # Scripts y datos para ejecución del seleccionador
└── 07_database_eficiencia     # Datos para medir eficiencia
    ├── audio                  # Datos de audio para eficiencia
    └── fotos                  # Datos de fotos para eficiencia
```

---

## Requisitos

- **Python 3.6** o superior
- Paquetes necesarios:

  ```bash
  pip install numpy scipy scikit-learn opencv-python matplotlib librosa
  ```

---

## Uso

1. **Ejecutar el programa principal**:

   ```bash
   python main.py
   ```

2. **Interacción con el Menú**:

   El programa presenta un menú con opciones para:

   - Analizar nuevamente la base de datos de imágenes o audios.
   - Mostrar la segregación (distribución) de imágenes.
   - Medir la eficiencia de los clasificadores.
   - Ejecutar el seleccionador de verduras.
   - Salir del programa.

3. **Analizar Bases de Datos**:

   - Seleccione la opción correspondiente para procesar las imágenes o audios.
   - El programa cargará, procesará y extraerá características de los datos, entrenando los clasificadores.

4. **Medir Eficiencia**:

   - Permite evaluar el rendimiento de los clasificadores con conjuntos de prueba.
   - Muestra la eficiencia porcentual y detalla los casos incorrectos.

5. **Ejecutar Seleccionador de Verduras**:

   - Se le solicitará que seleccione imágenes y un audio.
   - El programa clasificará las imágenes y el audio, mostrando las correspondencias y visualizaciones.

---

## Funciones Implementadas

- **`cargar_bases_datos()`**: Carga los datos desde archivos CSV y entrena los clasificadores.
- **`analizar_imagenes()`**: Analiza y procesa la base de datos de imágenes.
- **`analizar_audios()`**: Analiza y procesa la base de datos de audios.
- **`medir_eficiencia_imagenes()`**: Mide la eficiencia del clasificador de imágenes.
- **`medir_eficiencia_audios()`**: Mide la eficiencia del clasificador de audios.
- **`ejecutar_seleccionador()`**: Ejecuta el seleccionador que asocia imágenes con audios.

---

## Contribuciones

¡Las contribuciones son bienvenidas! Por favor, siga estos pasos:

1. Haga un fork del proyecto.
2. Cree una rama para su característica o corrección:

   ```bash
   git checkout -b feature/nueva-caracteristica
   ```

3. Realice los cambios necesarios y haga commit:

   ```bash
   git commit -m 'Añadir nueva característica'
   ```

4. Envíe los cambios a su fork:

   ```bash
   git push origin feature/nueva-caracteristica
   ```

5. Cree un Pull Request en este repositorio.

---

**Nota**: Este proyecto es una implementación educativa y puede requerir ajustes adicionales para adaptarse a entornos específicos o conjuntos de datos diferentes.
