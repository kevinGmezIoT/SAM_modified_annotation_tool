import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import os
import json
import argparse
from collections import deque

class ImageProcessor:
    def __init__(self, image, sam, label_info, prev_mask=None, prev_points=None):
        self.input_point = []
        self.input_label = []
        self.input_boxes = []
        self.color_list = []
        self.image = image
        self.show_image = image.copy()
        self.label_info = label_info
        self.label_ids = list(label_info.keys())
        
        # Initialize mask with previous frame's mask if available
        if prev_mask is not None and prev_mask.ndim == 2:
            # Convert previous mask to the right format (0 for background, class_id for objects)
            self.mask = [np.where(prev_mask == class_id, class_id, 0).astype(np.uint8) 
                        for class_id in range(max(self.label_ids) + 1)]
            print(f"Máscara anterior cargada con {np.count_nonzero(prev_mask)} píxeles")
        else:
            # Initialize empty masks for each class
            self.mask = [np.zeros_like(image[:,:,0], dtype=np.uint8) 
                        for _ in range(max(self.label_ids) + 1)]
            print("Inicializando máscaras vacías")
        
        # Initialize the current mask for the selected class
        self.current_mask = None
        self.show_mask_overlay = True  # Control para mostrar/ocultar la máscara
        
        # Store previous points for propagation
        self.prev_points = prev_points if prev_points is not None else []
        if self.prev_points:
            self.input_point = [p[0] for p in self.prev_points]
            self.input_label = [p[1] for p in self.prev_points]
            # Verde para puntos positivos (1), rojo para negativos (0)
            self.color_list = [(0, 255, 0) if label == 1 else (0, 0, 255) 
                             for label in self.input_label]
        
        # Initialize SAM predictor with the current image
        self.predictor = SamPredictor(sam)
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Default to first class (person)
        self.selected_class = self.label_ids[0]
        
        # UI settings
        self.fixed_window_size = (1000, 800)
        self.button_height = 30  # Smaller header for single class
        self.point_size = int((max(image.shape[1], image.shape[0])) * 0.005)
        
        # Track previous points for temporal consistency
        self.previous_points = deque(maxlen=10)  # Keep last 10 points

    def get_scaling_factors(self):
        """Calculate scaling factors between displayed image and original image"""
        if not hasattr(self, 'display_size') or self.display_size is None:
            return 1.0, 1.0
            
        h, w = self.image.shape[:2]
        disp_h, disp_w = self.display_size
        
        # Calculate the scaling factors
        scale_x = w / disp_w
        scale_y = h / disp_h
        
        return scale_x, scale_y

    def update_display(self):
        """Update the display with current points and mask"""
        if self.show_mask_overlay:
            display_image = self.show_image.copy()
        else:
            display_image = self.image.copy()
            
        # Draw points
        for point, color in zip(self.input_point, self.color_list):
            cv2.circle(display_image, (point[0], point[1]), 5, color, -1)
        
        # Update the display
        self.show_image = display_image
        self.update_window(display_image)

    def click_event(self, event, x, y, flags, param):
        # Ajustar la coordenada y para tener en cuenta la barra de herramientas
        y_img = y - self.button_height
        
        # Verificar si el clic está dentro del área de la imagen
        if not hasattr(self, 'display_size') or self.display_size is None:
            return
            
        disp_w, disp_h = self.display_size
        
        # Verificar si el clic está dentro de la imagen mostrada
        if (y_img < 0 or y_img >= disp_h or 
            x < 0 or x >= disp_w):
            return  # Fuera de los límites de la imagen
            
        # Obtener el tamaño original de la imagen
        h, w = self.image.shape[:2]
        
        # Calcular las coordenadas en la imagen original
        scale_x = w / disp_w
        scale_y = h / disp_h
        
        # Mapear las coordenadas del clic a la imagen original
        x_orig = int(x * scale_x)
        y_orig = int(y_img * scale_y)
        
        # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
        x_orig = max(0, min(x_orig, w - 1))
        y_orig = max(0, min(y_orig, h - 1))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.brush_mode:
                # Modo pincel: añadir a la máscara
                if self.current_mask is None:
                    self.current_mask = np.zeros_like(self.image[:,:,0], dtype=np.uint8)
                cv2.circle(self.current_mask, (x_orig, y_orig), int(self.brush_radius * scale_x), 1, -1)
                # Actualizar la máscara mostrada
                self.show_mask(self.current_mask, self.selected_class)
            else:
                # Modo puntos: añadir punto positivo y actualizar máscara
                self.input_point.append([x_orig, y_orig])
                self.input_label.append(1)  # 1 for foreground
                self.color_list.append((0, 255, 0))  # Green for positive points
                # Actualizar la máscara automáticamente
                if len(self.input_point) > 0:  # Solo si hay puntos
                    self.execute_prediction()
                # Actualizar la visualización
                self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.brush_mode:
                # Modo pincel: quitar de la máscara
                if self.current_mask is not None:
                    cv2.circle(self.current_mask, (x_orig, y_orig), int(self.brush_radius * scale_x), 0, -1)
                    # Actualizar la máscara mostrada
                    self.show_mask(self.current_mask, self.selected_class)
            else:
                # Modo puntos: añadir punto negativo y actualizar máscara
                self.input_point.append([x_orig, y_orig])
                self.input_label.append(0)  # 0 for background
                self.color_list.append((0, 0, 255))  # Red for negative points
                # Actualizar la máscara automáticamente
                if len(self.input_point) > 0:  # Solo si hay puntos
                    self.execute_prediction()
                # Actualizar la visualización
                self.update_display()
        elif event == cv2.EVENT_RBUTTONDOWN and not self.brush_mode:
            self.input_point.append((x, y-self.button_height))
            self.input_label.append(0)
            self.color_list.append((255, 0, 0))
            print(f"[NEGATIVE] Point clicked at ({x}, {y-self.button_height})")
            self.draw_points()

        elif event == cv2.EVENT_MOUSEMOVE and self.brush_mode:
            if y > self.button_height:
                paint_brush = np.zeros_like(self.image, dtype=np.uint8)
                cv2.circle(paint_brush, (x,y-self.button_height), self.brush_radius, (128, 128, 128), -1)  # Draw a filled circle at the last clicked point

                class_id = self.selected_class
                color = self.label_info[class_id]['color']
                mask = self.mask[class_id]==class_id
                masked_img = np.where(mask[..., None], color, self.image).astype(np.uint8)
                # use `addWeighted` to blend the two images
                temp_image = cv2.addWeighted(self.image, 0.6,masked_img, 0.4, 0)
                self.show_image = cv2.addWeighted(temp_image, 0.6, paint_brush, 0.4, 0)

        elif event == cv2.EVENT_RBUTTONDOWN and self.brush_mode:

            paint_brush = np.zeros_like(self.image, dtype=np.uint8)
            cv2.circle(paint_brush, (x, y - self.button_height), self.brush_radius, (128, 128, 128),
                       -1)  # Draw a filled circle at the last clicked point
            del_mask = (paint_brush[:, :, 0] == 128) & (paint_brush[:, :, 1] == 128) & (paint_brush[:, :, 2] == 128)
            self.mask[self.selected_class][del_mask] = 0
            mask = self.mask[self.selected_class] == self.selected_class
            color = self.label_info[self.selected_class]['color']
            masked_img = np.where(mask[..., None], color, self.image).astype(np.uint8)
            # use `addWeighted` to blend the two images
            temp_image = cv2.addWeighted(self.image, 0.6, masked_img, 0.4, 0)
            self.show_image = cv2.addWeighted(temp_image, 0.6, paint_brush, 0.4, 0)


        elif event == cv2.EVENT_LBUTTONDOWN and self.brush_mode:

            paint_brush = np.zeros_like(self.image, dtype=np.uint8)
            cv2.circle(paint_brush, (x, y - self.button_height), self.brush_radius, (128, 128, 128),
                       -1)  # Draw a filled circle at the last clicked point
            del_mask = (paint_brush[:, :, 0] == 128) & (paint_brush[:, :, 1] == 128) & (paint_brush[:, :, 2] == 128)
            color = self.label_info[self.selected_class]['color']
            self.mask[self.selected_class][del_mask] = self.selected_class
            mask = self.mask[self.selected_class] == self.selected_class
            masked_img = np.where(mask[..., None], color, self.image).astype(np.uint8)
            # use `addWeighted` to blend the two images
            temp_image = cv2.addWeighted(self.image, 0.6, masked_img, 0.4, 0)
            self.show_image = cv2.addWeighted(temp_image, 0.6, paint_brush, 0.4, 0)


    def update_window(self, display_image):
        """Update the window with the current display image"""
        if not hasattr(self, 'display_size') or self.display_size is None:
            return
            
        # Get current window size
        window_size = cv2.getWindowImageRect('image')
        if window_size[2] <= 0 or window_size[3] <= 0:
            return
            
        max_width = window_size[2]
        max_height = window_size[3] - self.button_height
        
        # Calculate aspect ratio
        h, w = display_image.shape[:2]
        aspect_ratio = w / h
        
        # Calculate new size maintaining aspect ratio
        if aspect_ratio > 1:  # Width > Height
            new_w = min(max_width, int(max_height * aspect_ratio))
            new_h = int(new_w / aspect_ratio)
        else:  # Height >= Width
            new_h = min(max_height, int(max_width / aspect_ratio))
            new_w = int(new_h * aspect_ratio)
        
        # Update display size for coordinate mapping
        self.display_size = (new_w, new_h)
        
        # Resize the image
        resized_display = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black canvas
        combined_image = np.zeros((new_h + self.button_height, max_width, 3), dtype=np.uint8)
        
        # Add button bar
        button_bar = np.zeros((self.button_height, max_width, 3), dtype=np.uint8)
        status_text = f"Labeling: {self.label_info[self.selected_class]['name']} | 's'=process | 'm'=brush | 'v'=ver máscara | 'r'=reset | 'a'=nuevo objeto | 'q'=guardar"
        cv2.putText(button_bar, status_text, (10, self.button_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine button bar and resized image
        combined_image[:self.button_height, :] = button_bar
        combined_image[self.button_height:self.button_height+new_h, :new_w] = resized_display
        
        cv2.imshow('image', combined_image)

    def show_mask(self, mask, class_id=None):
        if class_id is None:
            class_id = self.selected_class

        if class_id is not None:
            color = self.label_info[class_id]['color']
        else:
            color = np.array([0, 0, 128], dtype=np.uint8)  # Default color (blue)

        # Ensure mask is boolean
        mask = mask.astype(bool)
        
        # Store the current mask
        self.current_mask = mask
        print(f"Máscara almacenada con {np.count_nonzero(mask)} píxeles True")
        
        # Update the mask for the current class
        if class_id is not None and isinstance(self.mask, list):
            # Only update where the mask is True, keep existing values elsewhere
            self.mask[class_id] = np.where(mask, class_id, self.mask[class_id])
        
        # Create visualization with the class color
        masked_img = np.where(mask[..., None], color, self.image).astype(np.uint8)
        self.show_image = cv2.addWeighted(self.image, 0.6, masked_img, 0.4, 0)
        print(f"Máscara actualizada para clase {class_id}")

    def draw_points(self):
        for i,point in enumerate(self.input_point):
            color = self.color_list[i]
            cv2.circle(self.show_image, point, self.point_size+int(max(self.point_size*0.1, 2)), (255,255,255), -1)
            cv2.circle(self.show_image, point, self.point_size, color, -1)

    def execute_prediction(self):
        try:
            # Convert points to numpy arrays
            points = np.array(self.input_point)
            labels = np.array(self.input_label)
            
            # If we have previous points, add them with lower weight
            if len(self.previous_points) > 0:
                prev_points, prev_labels = zip(*self.previous_points)
                points = np.vstack([points, prev_points]) if len(points) > 0 else np.array(prev_points)
                labels = np.concatenate([labels, prev_labels]) if len(labels) > 0 else np.array(prev_labels)
            
            # Get the mask prediction
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,  # Only return the best mask
            )
            
            # Update the mask for the selected class
            if len(masks) > 0:
                self.show_mask(masks[0], self.selected_class)
                
        except Exception as e:
            print(f"Error during prediction: {e}")

    def enter_class_id(self):
        class_id = input("Enter class number ID: ")
        return int(class_id)

    def update_brush_radius(self, value):
        self.brush_radius = value

    def process_image(self):
        # Crear ventana y maximizarla
        cv2.namedWindow('image', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Obtener el tamaño de la pantalla
        screen_w = 1920  # Valores por defecto en caso de no poder obtener el tamaño real
        screen_h = 1080
        try:
            screen_w = cv2.getWindowImageRect('image')[2]
            screen_h = cv2.getWindowImageRect('image')[3]
        except:
            pass
            
        # Configurar el tamaño inicial de la ventana manteniendo la relación de aspecto
        h, w = self.image.shape[:2]
        aspect_ratio = w / h
        
        # Calcular el tamaño máximo que cabe en la pantalla manteniendo la relación de aspecto
        max_height = screen_h - 100  # Dejar espacio para la barra de tareas
        max_width = screen_w - 100   # Dejar un pequeño margen
        
        if aspect_ratio > 1:  # Ancho > Alto
            new_w = min(max_width, int(max_height * aspect_ratio))
            new_h = int(new_w / aspect_ratio)
        else:  # Alto > Ancho o igual
            new_h = min(max_height, int(max_width / aspect_ratio))
            new_w = int(new_h * aspect_ratio)
        
        # Redimensionar la ventana
        cv2.resizeWindow('image', new_w, new_h + self.button_height)
        
        # Mover la ventana a la esquina superior izquierda
        cv2.moveWindow('image', 0, 0)
            
        cv2.setMouseCallback('image', self.click_event)
        
        # Create a simple status bar
        button_image = np.zeros((self.button_height, self.image.shape[1], 3), dtype=np.uint8)
        status_text = f"Labeling: {self.label_info[self.selected_class]['name']} | 'm'=brush | 'v'=ver máscara | 'r'=reset | 'a'=nuevo objeto | 'q'=guardar"
        cv2.putText(button_image, status_text, (10, self.button_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Initialize brush settings
        self.brush_mode = False
        self.brush_radius = 30  # Default brush size
        cv2.createTrackbar('Brush Size', 'image', self.brush_radius, 100, self.update_brush_radius)
        
        # Si hay puntos guardados de la imagen anterior, generar automáticamente la máscara
        if self.prev_points and len(self.prev_points) > 0:
            print(f"Generando máscara automáticamente con {len(self.prev_points)} puntos guardados")
            try:
                # Ejecutar la predicción con los puntos guardados
                self.execute_prediction()
                print("Máscara generada exitosamente para la nueva imagen")
            except Exception as e:
                print(f"Error generando máscara automática: {e}")
                # Si falla, mostrar solo los puntos
                self.update_display()
        else:
            # Si no hay puntos previos, solo mostrar la imagen
            self.show_image = self.image.copy()

        while True:
            # Display the current image with UI elements
            if self.show_mask_overlay:
                display_image = self.show_image
            else:
                display_image = self.image.copy()
                # Mostrar solo los puntos sin la máscara
                for point, label in zip(self.input_point, self.input_label):
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)
                    cv2.circle(display_image, (int(point[0]), int(point[1])), 5, color, -1)
            
            # Obtener el tamaño actual de la ventana
            window_size = cv2.getWindowImageRect('image')
            if window_size[2] > 0 and window_size[3] > 0:  # Si la ventana es válida
                # Calcular el tamaño manteniendo la relación de aspecto
                h, w = display_image.shape[:2]
                aspect_ratio = w / h
                
                # Calcular el nuevo tamaño manteniendo la relación de aspecto
                max_width = window_size[2]
                max_height = window_size[3] - self.button_height  # Restar la altura de la barra de botones
                
                if aspect_ratio > 1:  # Ancho > Alto
                    new_w = min(max_width, int(max_height * aspect_ratio))
                    new_h = int(new_w / aspect_ratio)
                else:  # Alto > Ancho o igual
                    new_h = min(max_height, int(max_width / aspect_ratio))
                    new_w = int(new_h * aspect_ratio)
                
                # Guardar el tamaño de visualización actual para el mapeo de coordenadas
                self.display_size = (new_w, new_h)
                
                # Redimensionar la imagen manteniendo la relación de aspecto
                resized_display = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Crear una imagen con fondo negro del tamaño de la ventana
                combined_image = np.zeros((new_h + self.button_height, max_width, 3), dtype=np.uint8)
                
                # Copiar la barra de botones y la imagen redimensionada
                combined_image[:self.button_height, :] = cv2.resize(button_image, (max_width, self.button_height))
                combined_image[self.button_height:self.button_height+new_h, :new_w] = resized_display
                
                cv2.imshow('image', combined_image)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Process the current points
                if len(self.input_point) > 0:
                    self.execute_prediction()
                    # Add points to history for temporal consistency
                    for point, label in zip(self.input_point, self.input_label):
                        self.previous_points.append((point, label))
                else:
                    print("Add points first (left click for foreground, right click for background)")
                    
            elif key == ord('a'):  # Add new object
                print("Adding a new object")
                self.input_point = []
                self.input_label = []
                
            elif key == ord('m'):  # Toggle brush mode
                self.brush_mode = not self.brush_mode
                if not self.brush_mode:
                    # Update the mask display when exiting brush mode
                    mask = self.mask[self.selected_class] == self.selected_class
                    color = self.label_info[self.selected_class]['color']
                    masked_img = np.where(mask[..., None], color, self.image).astype(np.uint8)
                    self.show_image = cv2.addWeighted(self.image, 0.6, masked_img, 0.4, 0)
                else:
                    print("Brush mode: Left click to add, Right click to remove")

            elif key == ord('r'):  # Resetear todo
                self.input_point = []
                self.input_label = []
                self.color_list = []
                self.current_mask = None
                self.show_image = self.image.copy()
                print("Puntos y máscara reiniciados")
                
            elif key == ord('v'):  # Alternar visualización de la máscara
                self.show_mask_overlay = not self.show_mask_overlay
                print(f"Visualización de máscara: {'activada' if self.show_mask_overlay else 'desactivada'}")
                
            elif key == ord('q'):
                if self.current_mask is not None and isinstance(self.current_mask, np.ndarray):
                    # Asegurar que current_mask es booleano
                    mask_bool = self.current_mask.astype(bool)
                    
                    print(f"Procesando máscara con {np.count_nonzero(mask_bool)} píxeles True")
                    print(f"Clase seleccionada: {self.selected_class}")
                    
                    # Crear la máscara final directamente con los valores correctos
                    # La máscara debe tener el valor de la clase donde es True, y 0 donde es False
                    final_mask = np.zeros_like(self.image[:,:,0], dtype=np.uint8)
                    final_mask[mask_bool] = self.selected_class
                    
                    print(f"Máscara final creada con {np.count_nonzero(final_mask)} píxeles")
                    print(f"Valores únicos en final_mask: {np.unique(final_mask)}")
                    
                    # Si la clase seleccionada es 0, necesitamos usar un valor diferente para la máscara
                    # porque 0 se usa para el fondo
                    if self.selected_class == 0:
                        # Usar 255 para la máscara binaria en lugar del valor de clase 0
                        self.mask = np.where(mask_bool, 255, 0).astype(np.uint8)
                    else:
                        self.mask = final_mask
                    
                    print(f"Guardando máscara con {np.count_nonzero(self.mask)} píxeles")
                    print(f"Valores únicos en la máscara guardada: {np.unique(self.mask)}")
                    
                    # Guardar los puntos actuales para el siguiente frame
                    current_points = list(zip(self.input_point, self.input_label))
                else:
                    print("No hay máscara válida para guardar")
                    self.mask = None
                    current_points = []
                break

        cv2.destroyAllWindows()

        return self.image, self.mask, current_points


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A script with string command-line parameters.')

    # Adding a required string parameter
    parser.add_argument('config_path', type=str, help='Path to config file.')

    args = parser.parse_args()

    config_path = args.config_path

    # Read configuration from JSON file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    sam_model_name = config["sam_model"]["name"]
    checkpoint_path = config["sam_model"]["checkpoint_path"]
    sam = sam_model_registry[sam_model_name](checkpoint=checkpoint_path).cuda()

    # Convert string keys to integers for label_info dictionary
    label_info = {int(key): value for key, value in config["label_info"].items()}

    raw_data = config["raw_data_path"]
    out_path = config["output_path"]["root"]

    img_out_path = config["output_path"]["img_subpath"]
    label_out_path = config["output_path"]["label_subpath"]
    img_out_path = os.path.join(out_path, img_out_path)
    label_out_path = os.path.join(out_path, label_out_path)

    max_dimension = int(config["max_image_dimension"])

    # Get a list of all files in the folder
    files = os.listdir(raw_data)

    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    # Create a list to store the paths of image files
    image_paths = []

    # Iterate through the files and add image file paths to the list
    for file in files:
        # Check if the file has a valid image extension
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(raw_data, file).replace("\\","/"))

    # Check if the folder exists
    if not os.path.exists(out_path):
        # If it doesn't exist, create it
        os.mkdir(out_path)
        os.mkdir(img_out_path)
        os.mkdir(label_out_path)
        print(f"Folder '{out_path}' created successfully.")
        index = -1
    else:
        print(f"Folder '{out_path}' already exists.")
        # Ask the user if they want to resume a previous labeling
        resume_labeling = input("Do you want to resume a previous labeling? (y/n) ").strip().lower()

        if resume_labeling == 'y':
            last_image = input("Please insert thqe last labelled image: ").strip()

            # Split the file name and extension
            _, subfix = os.path.splitext(image_paths[0])
            prefix, _ = os.path.splitext(last_image)

            last_image = prefix+subfix

            last_image = os.path.join(config["raw_data_path"], last_image)
            try:
                index = image_paths.index(last_image)
                print(f"The index of '{last_image}' in image_paths is: {index}")
                print(f"Restarting from image {image_paths[index + 1]}")
            except ValueError:
                print(f"'{last_image}' is not in the image_paths list. Restart from 0")
                index = -1
        else:
            index = -1
            print("Starting a new labeling session.")
            print("\033[91m***[WARNING]*** CURRENT LABELS WILL BE OVERWRITTEN.\033[0m")

    print("=== People Labeling Tool ===")
    print("1. Click on the person to add positive points (left click) or negative points (right click)")
    print("2. Press 's' to process the points and update the mask")
    print("3. Use 'm' to toggle brush mode for fine-tuning:")
    print("   - Left click to add to the mask")
    print("   - Right click to remove from the mask")
    print("4. Press 'a' to start a new object")
    print("5. Press 'q' to save and move to the next image")
    print("\nTip: The mask from the previous frame is automatically used as a starting point.")
    print("      This makes labeling sequential frames much faster!\n")

    # Process images sequentially with mask and points propagation
    prev_mask = None
    prev_points = None
    
    for idx, image_path in enumerate(image_paths[index+1:]):
        print(f"Processing image: {image_path}, {idx+1} of {len(image_paths)}")
        print("Press 'q' to save and continue, 's' to process points, 'm' to toggle brush mode")
        
        out_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        
        # If we have a previous mask, use it to initialize the current frame
        if prev_mask is not None:
            print("Using previous frame's mask and points as starting point")
            # Resize the previous mask to match the current frame if needed
            if prev_mask.shape != image.shape[:2]:
                prev_mask = cv2.resize(prev_mask, (image.shape[1], image.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Process the current frame with the previous mask and points

        # Resize the image if either dimension is larger than 1024
        if image.shape[0] > max_dimension or image.shape[1] > max_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            aspect_ratio = image.shape[1] / image.shape[0]
            new_height = min(max_dimension, int(max_dimension / aspect_ratio))
            new_width = min(max_dimension, int(aspect_ratio * max_dimension))
            image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

        img_proc = ImageProcessor(image, sam, label_info, prev_mask, prev_points)
        image, mask, prev_points = img_proc.process_image()
        prev_mask = mask  # Update the previous mask for the next frame

        if mask is None:
            print("No mask created, skipping...")
            continue
            
        # Update the previous mask for the next frame
        prev_mask = mask

        # Save the data
        image_filename = out_name+".png"  # Specify the filename for the image
        image_save_path = os.path.join(img_out_path, image_filename)
        cv2.imwrite(image_save_path, image)
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        # Guardar la máscara binaria
        label_save_path = os.path.join(label_out_path, image_filename)
        cv2.imwrite(label_save_path, binary_mask)

        # Concatenate the image and mask horizontally
        combined_image = np.hstack((image, cv2.cvtColor(mask*25, cv2.COLOR_GRAY2BGR)))

        # Display the combined image in a single window
        cv2.imshow("SAVED Image and Mask", combined_image)
        cv2.waitKey(1000)  # Wait for 1 second (1000 milliseconds)
        cv2.destroyAllWindows()






