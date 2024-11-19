import cv2
import numpy as np
from kmeans import KMeans

class Imagen:
    def __init__(self):
        self.image = None
        self.contours = None
        self.masked_image = None
        self.mean_color = None
        self.hu_moments = None


    def imagen_desde_archivo(self, image_path):
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError(f"Error: No se pudo cargar la imagen {image_path}.")
        except Exception as e:
            print(f"Error: {e}")
            self.image = None


    def mask_image_kmeans(self, n_clusters):
        # Validate n_clusters
        if n_clusters not in [2, 3, 4]:
            raise ValueError("n_clusters must be 2, 3, or 4")

        # Resize image to 720px height while maintaining aspect ratio
        h, w = self.image.shape[:2]
        target_height = 720
        target_width = int((target_height * w) / h)
        self.resized_image = cv2.resize(self.image, (target_width, target_height))

        # Convert to float32 for faster computations
        reshaped_image = self.resized_image.reshape(-1, 3).astype(np.float32)

        # Compute average color of central rectangle
        grid_size = 11
        central_rect_height = target_height // grid_size
        central_rect_width = target_width // grid_size

        y_start = (target_height - central_rect_height) // 2
        y_end = y_start + central_rect_height
        x_start = (target_width - central_rect_width) // 2
        x_end = x_start + central_rect_width

        central_rect = self.resized_image[y_start:y_end, x_start:x_end]
        average_color_central = np.mean(central_rect.reshape(-1, 3), axis=0)

        # Define average colors (BGR format)
        average_blue = np.array([255, 0, 0], dtype=np.float32)
        average_white = np.array([255, 255, 255], dtype=np.float32)
        average_gray = np.array([127.5, 127.5, 127.5], dtype=np.float32)

        # Initialize centroids based on n_clusters
        if n_clusters == 2:
            initial_centroids = np.array([average_blue, average_color_central], dtype=np.float32)
        elif n_clusters == 3:
            initial_centroids = np.array([average_white, average_gray, average_color_central], dtype=np.float32)
        elif n_clusters == 4:
            average_black = np.array([0, 0, 0], dtype=np.float32)
            initial_centroids = np.array([average_white, average_gray, average_color_central, average_black], dtype=np.float32)

        # K-means clustering with retry mechanism
        max_attempts = 10
        total_points = reshaped_image.shape[0]
        for attempt in range(max_attempts):
            # Initialize KMeans
            kmeans = KMeans(n_clusters=n_clusters, max_iters=1000, tol=1e-3)
            kmeans.centroids = initial_centroids.copy()
            kmeans.fit(reshaped_image)

            # Check cluster sizes
            labels = kmeans.labels
            counts = np.bincount(labels)
            max_cluster_size = np.max(counts)

            if max_cluster_size / total_points <= 0.95:
                # Acceptable clustering
                break
            else:
                # Perturb initial centroids slightly and retry
                perturbation = np.random.uniform(-15, 15, initial_centroids.shape)
                initial_centroids = initial_centroids + perturbation
        else:
            raise RuntimeError("Failed to cluster without exceeding cluster size limit after multiple attempts.")

        # Reconstruct image with clusters
        self.kmeans_image = kmeans.centroids[kmeans.labels].reshape(target_height, target_width, 3).astype(np.uint8)

        # Calculate distances using only the blue channel
        distances_to_blue = np.abs(kmeans.centroids[:, 0] - average_blue[0])

        # Determine centroids to keep based on n_clusters
        if n_clusters == 2:
            # Keep the centroid farthest from average blue
            target_centroid_index = np.argmax(distances_to_blue)
            target_centroid_indices = [target_centroid_index]
        elif n_clusters == 3:
            # Eliminate centroids closest to white and gray
            distances_to_white = np.linalg.norm(kmeans.centroids - average_white, axis=1)
            distances_to_gray = np.linalg.norm(kmeans.centroids - average_gray, axis=1)
            summed_distances = distances_to_white + distances_to_gray
            target_centroid_index = np.argmax(summed_distances)
            target_centroid_indices = [target_centroid_index]
        elif n_clusters == 4:
            # Keep the two centroids farthest from white and gray
            distances_to_white = np.linalg.norm(kmeans.centroids - average_white, axis=1)
            distances_to_gray = np.linalg.norm(kmeans.centroids - average_gray, axis=1)
            summed_distances = distances_to_white + distances_to_gray
            target_centroid_indices = np.argsort(summed_distances)[-2:]

        # Initialize variables
        binary = np.zeros((target_height, target_width), dtype=np.uint8)
        contour_found = False

        # Iterate over selected centroids
        for target_centroid_index in target_centroid_indices:
            # Create mask for the current centroid
            mask = kmeans.labels == target_centroid_index
            binary.fill(0)  # Reset binary image
            binary[mask.reshape(target_height, target_width)] = 255

            # Detect contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Check each contour
            for cnt in contours:
                if not self.is_contour_touching_edge(cnt, target_width, target_height):
                    # Suitable contour found
                    max_contour = cnt
                    contour_found = True
                    break  # Exit contour loop

            if contour_found:
                break  # Exit centroid loop

        if contour_found:
            self.small_contour = max_contour

            # Scale contour to original size
            scale_x = w / target_width
            scale_y = h / target_height
            scaled_contour = (max_contour * [scale_x, scale_y]).astype(np.int32)
            self.contours = scaled_contour

            # Create final mask in original size
            final_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(final_mask, [scaled_contour], -1, 255, -1)

            # Convert final_mask to 3-channel image
            final_mask_rgb = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

            # Apply the mask
            self.masked_image = cv2.bitwise_and(self.image, final_mask_rgb)
        else:
            self.contours = None
            self.masked_image = np.zeros_like(self.image)

    def is_contour_touching_edge(self, contour, width, height):
        # Check if any point in the contour touches the edge
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        return np.any(x <= 0) or np.any(x >= width - 1) or np.any(y <= 0) or np.any(y >= height - 1)


    def extraer_caracteristicas(self):
        mask = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY) > 0
        mean_color = cv2.mean(self.masked_image, mask=mask.astype(np.uint8))
        gray_masked_image = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY)
        gray_masked_image[gray_masked_image > 0] = 255
        moments = cv2.moments(gray_masked_image)
        hu_moments = cv2.HuMoments(moments).flatten()
        log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        self.log_hu_moments = log_hu_moments
        self.mean_color = mean_color[:3]

    def procesar_imagen_individual(self):
        if self.image is None:
            return None

        self.mask_image_kmeans(2)
        self.extraer_caracteristicas()
        # self.mostrar_proceso()

        # Crear vector de características combinando color y forma
        features = np.hstack((
            self.mean_color,
            self.log_hu_moments[1],
        ))

        self.features = features
    


    # Metodos de debug
    
    def mostrar_proceso(self):
        # 1. Imagen original redimensionada
        resized = self.resized_image.copy()
        
        # 2. Imagen con kmeans y grid de 7x7
        target_height, target_width = self.kmeans_image.shape[:2]
        central_rect_height = target_height // 7
        central_rect_width = target_width // 7
        x_start = (target_width - central_rect_width) // 2
        x_end = x_start + central_rect_width
        y_start = (target_height - central_rect_height) // 2
        y_end = y_start + central_rect_height

        kmeans_vis = self.resized_image.copy()
        for i in range(0, target_height, central_rect_height):
            cv2.line(kmeans_vis, (0, i), (target_width, i), (255, 0, 0), 1)
        for j in range(0, target_width, central_rect_width):
            cv2.line(kmeans_vis, (j, 0), (j, target_height), (255, 0, 0), 1)
        cv2.rectangle(kmeans_vis, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        
        # 3. Imagen kmeans con contorno
        kmeans_contour = self.kmeans_image.copy()
        if hasattr(self, 'small_contour'):
            cv2.drawContours(kmeans_contour, [self.small_contour], -1, (0, 255, 0), 2)
        
        # 4. Imagen original con fondo recortado
        final_result = self.masked_image.copy()
        
        # Mostrar las 4 imágenes
        images = [resized, kmeans_vis, kmeans_contour, final_result]
        resized_images = [self.resize_image(img, 300, 400) for img in images]
        stacked_image = np.hstack(resized_images)
        
        cv2.imshow('Proceso de Imagen', stacked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


