# Deteksi Kontur Wajah dan Ekspresi dengan MediaPipe

Program ini menggunakan MediaPipe Face Mesh untuk mendeteksi kontur wajah dan menganalisis ekspresi wajah secara real-time.

## Fitur

- **Deteksi Kontur Wajah**: Mendeteksi 468 landmark wajah menggunakan MediaPipe Face Mesh
- **Analisis Ekspresi**: Mendeteksi berbagai ekspresi wajah:
  - Netral
  - Senang
  - Sedih
  - Marah
  - Terkejut
  - Mulut Terbuka
- **Real-time Processing**: Memproses video dari webcam secara real-time
- **Visualisasi**: Menampilkan kontur wajah dan informasi metrik

## Metrik yang Digunakan

1. **EAR (Eye Aspect Ratio)**: Mengukur seberapa terbuka mata
2. **MAR (Mouth Aspect Ratio)**: Mengukur seberapa terbuka mulut
3. **Eyebrow Position**: Mengukur posisi alis relatif terhadap mata

## Instalasi

### 1. Clone atau Download Project

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Atau install secara manual:

```bash
pip install opencv-python mediapipe numpy
```

## Cara Menggunakan

### Menjalankan dengan Webcam

```bash
python face_expression_detection.py
```

### Menjalankan dengan Video File

Modifikasi bagian terakhir dalam file `face_expression_detection.py`:

```python
if __name__ == "__main__":
    detector = FaceExpressionDetector()
    detector.run(source="path/to/video.mp4")  # Ganti dengan path video Anda
```

## Kontrol

- **'q'**: Keluar dari program

## Struktur Kode

### Class `FaceExpressionDetector`

#### Methods Utama:

- `__init__()`: Inisialisasi MediaPipe Face Mesh dan mendefinisikan landmark indices
- `calculate_ear()`: Menghitung Eye Aspect Ratio untuk deteksi mata
- `calculate_mar()`: Menghitung Mouth Aspect Ratio untuk deteksi mulut
- `calculate_eyebrow_position()`: Menghitung posisi alis
- `detect_expression()`: Mendeteksi ekspresi berdasarkan metrik
- `draw_landmarks()`: Menggambar landmark pada frame
- `process_frame()`: Memproses setiap frame untuk deteksi
- `run()`: Menjalankan loop utama program

## Cara Kerja

1. **Deteksi Wajah**: MediaPipe Face Mesh mendeteksi wajah dan 468 landmark
2. **Ekstraksi Fitur**: Program mengekstrak fitur penting (mata, alis, mulut)
3. **Kalkulasi Metrik**: 
   - EAR dihitung dari jarak vertikal dan horizontal mata
   - MAR dihitung dari rasio jarak vertikal dan horizontal mulut
   - Posisi alis dihitung dari jarak alis ke mata
4. **Klasifikasi Ekspresi**: Ekspresi ditentukan berdasarkan threshold metrik
5. **Visualisasi**: Landmark dan informasi ditampilkan pada frame

## Threshold Ekspresi

- **Terkejut**: EAR > 0.25, MAR > 0.5, Eyebrow < -0.02
- **Senang**: MAR > 0.3 dan < 0.5, EAR > 0.2
- **Sedih**: EAR < 0.2, MAR < 0.2
- **Marah**: Eyebrow > -0.01, EAR > 0.18, MAR < 0.3
- **Mulut Terbuka**: MAR > 0.5

## Persyaratan Sistem

- Python 3.7 atau lebih tinggi
- Webcam (untuk deteksi real-time)
- Windows/Linux/MacOS

## Troubleshooting

### Kamera Tidak Terdeteksi

Jika kamera tidak terdeteksi, coba ganti `source=0` dengan `source=1` atau nomor lain:

```python
detector.run(source=1)
```

### Performance Lambat

Jika program berjalan lambat:
1. Turunkan resolusi kamera
2. Kurangi `max_num_faces` menjadi 1
3. Nonaktifkan visualisasi mesh lengkap

## Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan:

1. **Machine Learning**: Gunakan model ML untuk klasifikasi ekspresi yang lebih akurat
2. **Logging**: Simpan data ekspresi ke file untuk analisis
3. **Multi-face**: Deteksi ekspresi untuk beberapa wajah sekaligus
4. **Custom Expressions**: Tambahkan ekspresi custom sesuai kebutuhan
5. **GUI**: Buat interface grafis dengan Tkinter atau PyQt

## Referensi

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Eye Aspect Ratio for Blink Detection](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

## Lisensi

MIT License

## Author

Riset Face Contour - 2025
