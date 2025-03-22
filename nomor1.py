import cv2
import numpy as np
import matplotlib.pyplot as plt

def konvolusi(image, mask):
    """ 
    Fungsi untuk melakukan konvolusi pada gambar warna dengan kernel 3x3. 
    """ 
    N, M, C = image.shape  # Dapatkan ukuran gambar dan jumlah channel warna (RGB)
    ImageResult = np.zeros((N, M, C), dtype=np.uint8)  # hasil konvolusi

    # Loop sesuai kode C
    for i in range(1, N - 2):
        for j in range(1, M - 2):
            for c in range(C):  # Loop untuk tiap channel warna (R, G, B)
                ImageResult[i, j, c] = (
                    image[i - 1, j - 1, c] * mask[0][0] +
                    image[i - 1, j, c] * mask[0][1] +
                    image[i - 1, j + 1, c] * mask[0][2] +
                    image[i, j - 1, c] * mask[1][0] +
                    image[i, j, c] * mask[1][1] +
                    image[i, j + 1, c] * mask[1][2] +
                    image[i + 1, j - 1, c] * mask[2][0] +
                    image[i + 1, j, c] * mask[2][1] +
                    image[i + 1, j + 1, c] * mask[2][2]
                )

    # Pastikan hasil tetap dalam rentang 0-255
    ImageResult = np.clip(ImageResult, 0, 255)
    return ImageResult.astype(np.uint8)

# Membaca gambar berwarna
image = cv2.imread("D:\#Praktikum6\latihanOpencv\Dataset\maruko2.png")

# Pastikan gambar berhasil dimuat
if image is None:
    print("Error: Gambar tidak ditemukan!")
else:
    # Konversi BGR ke RGB agar tampil dengan benar di Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Definisi berbagai kernel
    kernels = [
        ("Kernel A", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int32)),
        ("Kernel B", np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.int32)),
        ("Kernel C", np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.int32)),
        ("Kernel D", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int32)),
        ("Kernel E", np.array([[0.5, -1, 0.5], [-1, 0.5, -1], [0.5, -1, 0.5]], dtype=np.float32)),
        ("Kernel F", np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32))
    ]

    # Terapkan konvolusi pada gambar warna dengan berbagai kernel
    results = [(title, konvolusi(image_rgb, kernel)) for title, kernel in kernels]

    # Tampilkan gambar asli dan hasil konvolusi dalam satu tab
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Gambar Asli")
    axes[0, 0].axis("off")

# Tampilkan hasil konvolusi di posisi lainnya
for idx, (title, result) in enumerate(results):
    if idx < 3:  # Baris atas (3 gambar)
        axes[0, idx + 1].imshow(result)
        axes[0, idx + 1].set_title(title)
        axes[0, idx + 1].axis("off")
    else:  # Baris bawah (4 gambar)
        axes[1, idx - 3].imshow(result)
        axes[1, idx - 3].set_title(title)
        axes[1, idx - 3].axis("off")

# Sembunyikan subplot yang tidak digunakan
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()