import cv2
import numpy as np
import os  # Import modul os untuk mengelola direktori

def konvolusi(image, mask):
    """
    Fungsi untuk melakukan konvolusi pada gambar grayscale dengan kernel 3x3.
    """
    N, M = image.shape  # Dapatkan ukuran gambar
    ImageResult = np.zeros((N, M), dtype=np.uint8)  # hasil konvolusi

    # Loop untuk konvolusi
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            ImageResult[i, j] = (
                image[i - 1, j - 1] * mask[0][0] +
                image[i - 1, j] * mask[0][1] +
                image[i - 1, j + 1] * mask[0][2] +
                image[i, j - 1] * mask[1][0] +
                image[i, j] * mask[1][1] +
                image[i, j + 1] * mask[1][2] +
                image[i + 1, j - 1] * mask[2][0] +
                image[i + 1, j] * mask[2][1] +
                image[i + 1, j + 1] * mask[2][2]
            )

    # Pastikan hasil tetap dalam rentang 0-255
    ImageResult = np.clip(ImageResult, 0, 255)
    return ImageResult.astype(np.uint8)

# Definisi berbagai kernel
kernels = [
    ("Original", None),  # Tanpa kernel (gambar asli)
    ("Kernel A (Tengah=1, Tetangga=0)", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    ("Kernel B (Tengah<0, Tetangga=0)", np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])),
    ("Kernel C (Baris 1<0, Tengah<=0, Lainnya>=1)", np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])),
    ("Kernel D (Tengah>1, Tetangga<1)", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    ("Kernel E (Tengah<1, Tetangga Bervariasi)", np.array([[0.5, -1, 0.5], [-1, 0.5, -1], [0.5, -1, 0.5]]))
]

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses!")
    exit()

# Buat folder "HasilSet" jika belum ada
output_folder = "HasilSet"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Counter untuk penamaan file screenshot
screenshot_count = 0

# Flag untuk menentukan apakah konvolusi harus dilakukan
apply_convolution = False

# Loop untuk menangkap frame secara real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal menangkap frame!")
        break

    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Jika tombol 's' ditekan, aktifkan flag untuk melakukan konvolusi
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        apply_convolution = True
        screenshot_count += 1

    # Jika flag aktif, lakukan konvolusi dan tampilkan hasilnya
    if apply_convolution:
        # List untuk menyimpan hasil konvolusi
        results = []

        # Terapkan konvolusi untuk setiap kernel
        for title, kernel in kernels:
            if kernel is None:
                result = gray  # Gambar asli (tanpa konvolusi)
            else:
                result = konvolusi(gray, kernel)
            
            # Tambahkan teks ke gambar
            result_with_text = cv2.putText(result.copy(), title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            results.append(result_with_text)

        # Gabungkan semua hasil secara horizontal
        combined_results = cv2.hconcat(results[:3])  # Baris pertama: 3 gambar
        combined_results2 = cv2.hconcat(results[3:])  # Baris kedua: 3 gambar
        final_output = cv2.vconcat([combined_results, combined_results2])  # Gabungkan kedua baris

        # Tampilkan hasil gabungan
        cv2.imshow("Hasil Konvolusi", final_output)

        # Simpan screenshot ke folder "HasilSet"
        screenshot_name = os.path.join(output_folder, f"hasil_{screenshot_count}.png")
        cv2.imwrite(screenshot_name, final_output)
        print(f"Screenshot disimpan sebagai {screenshot_name}")

        # Nonaktifkan flag setelah konvolusi selesai
        apply_convolution = False
    else:
        # Tampilkan gambar asli jika tombol 's' belum ditekan
        cv2.imshow("Hasil Konvolusi", gray)

    # Jika tombol 'q' ditekan, keluar dari loop
    if key == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()