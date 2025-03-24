import cv2
import numpy as np

# Load gambar dengan noise
img_path = "C:\\semester 6\\visi komputer\\tugasDiskusi4-Visikom\\DataSet\\polindra_noise.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Terapkan berbagai metode filtering
median_9 = cv2.medianBlur(img, 9)  # Median Filter 9x9
gaussian_9 = cv2.GaussianBlur(img, (9, 9), 0)  # Gaussian Filter 9x9
bilateral = cv2.bilateralFilter(img, 9, 75, 75)  # Bilateral Filter

# Simpan hasil filtering ke file
cv2.imwrite("C:\\semester 6\\visi komputer\\tugasDiskusi4-Visikom\\HasilSet\\hasil_median_9x9.jpg", median_9)
cv2.imwrite("C:\\semester 6\\visi komputer\\tugasDiskusi4-Visikom\\HasilSet\\hasil_gaus_9x9.jpg", gaussian_9)
cv2.imwrite("C:\\semester 6\\visi komputer\\tugasDiskusi4-Visikom\\HasilSet\\hasil_bilateral.jpg", bilateral)

print("Semua hasil filtering telah disimpan!")
