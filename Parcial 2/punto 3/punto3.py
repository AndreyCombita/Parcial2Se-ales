import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CARGA DE IMAGEN EN ESCALA DE GRISES
# ==========================================
nombre_imagen = 'imagen.jpg'  # Reemplaza por el nombre de tu imagen
imagen = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    raise FileNotFoundError(f"No se encontró la imagen '{nombre_imagen}'. Verifica la ruta.")

# ==========================================
# 2. DEFINICIÓN DE MATRICES / KERNELS
# ==========================================
# Kernel Pasa Bajas (Promediador 5x5 - Suma = 1)
kernel_pasa_bajas = np.ones((5, 5), dtype=np.float32) / 25.0

# Kernel Pasa Altas 00 (Suma = 1)
kernel_pasa_altas00 = np.array([
    [ 2,  2,  2,  2,  2],
    [ 2, -3, -3, -3,  2],
    [ 2, -3, -7, -3,  2],
    [ 2, -3, -3, -3,  2],
    [ 2,  2,  2,  2,  2]
], dtype=np.float32)

# Kernel Pasa Altas 02 (Suma = 0)
kernel_pasa_altas02 = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0, -3, -3, -3,  0],
    [ 0, -3, 24, -3,  0],
    [ 0, -3, -3, -3,  0],
    [ 0,  0,  0,  0,  0]
], dtype=np.float32)

# ==========================================
# 3. APLICACIÓN DE FILTROS 2D
# ==========================================
img_pasa_altas02 = cv2.filter2D(imagen, -1, kernel_pasa_altas02)
img_pasa_altas00 = cv2.filter2D(imagen, -1, kernel_pasa_altas00)
img_pasa_bajas = cv2.filter2D(imagen, -1, kernel_pasa_bajas)
img_pasa_bandas = cv2.filter2D(img_pasa_bajas, -1, kernel_pasa_altas02)

# ==========================================
# 4. GRAFICACIÓN Y GUARDADO DE RESULTADOS
# ==========================================
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_pasa_altas02, cmap='gray')
plt.title('1. Pasa Altas (Kernel Suma = 0)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_pasa_altas00, cmap='gray')
plt.title('2. Pasa Altas (Kernel Suma = 1)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_pasa_bajas, cmap='gray')
plt.title('3. Pasa Bajas (Promediador 5x5)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_pasa_bandas, cmap='gray')
plt.title('4. Pasa Bandas (Bajas + Altas)')
plt.axis('off')

plt.tight_layout()
plt.savefig("grafica_punto3.png", dpi=300)
plt.show()