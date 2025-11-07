import cv2
import mediapipe as mp
import numpy as np
import time

class FaceExpressionDetector:
    def __init__(self):
        # Inisialisasi MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Landmark indices untuk analisis ekspresi
        # Mata kiri
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        # Mata kanan
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Alis kiri
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        # Alis kanan
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336]
        # Mulut
        self.MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        # Dagu
        self.CHIN = [152]
        
    def calculate_ear(self, landmarks, eye_indices):
        """Menghitung Eye Aspect Ratio (EAR) untuk deteksi kedipan"""
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        
        # Jarak vertikal
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        
        # Jarak horizontal
        C = np.linalg.norm(points[0] - points[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, landmarks):
        """Menghitung Mouth Aspect Ratio (MAR) untuk deteksi mulut terbuka"""
        # Titik atas dan bawah mulut
        upper_lip = landmarks[13]  # Bibir atas tengah
        lower_lip = landmarks[14]  # Bibir bawah tengah
        
        # Titik kiri dan kanan mulut
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Jarak vertikal
        vertical_dist = np.sqrt((upper_lip.x - lower_lip.x)**2 + 
                               (upper_lip.y - lower_lip.y)**2)
        
        # Jarak horizontal
        horizontal_dist = np.sqrt((left_mouth.x - right_mouth.x)**2 + 
                                 (left_mouth.y - right_mouth.y)**2)
        
        # MAR formula
        mar = vertical_dist / horizontal_dist
        return mar
    
    def calculate_eyebrow_position(self, landmarks):
        """Menghitung posisi alis untuk deteksi ekspresi terkejut/marah"""
        # Alis kiri
        left_eyebrow_y = np.mean([landmarks[i].y for i in self.LEFT_EYEBROW])
        left_eye_y = np.mean([landmarks[i].y for i in self.LEFT_EYE])
        
        # Alis kanan
        right_eyebrow_y = np.mean([landmarks[i].y for i in self.RIGHT_EYEBROW])
        right_eye_y = np.mean([landmarks[i].y for i in self.RIGHT_EYE])
        
        # Jarak alis ke mata (nilai negatif berarti alis naik)
        left_distance = left_eyebrow_y - left_eye_y
        right_distance = right_eyebrow_y - right_eye_y
        
        avg_distance = (left_distance + right_distance) / 2
        return avg_distance
    
    def detect_expression(self, landmarks):
        """Mendeteksi ekspresi wajah berdasarkan landmark"""
        # Hitung metrik
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.calculate_mar(landmarks)
        eyebrow_pos = self.calculate_eyebrow_position(landmarks)
        
        # Deteksi ekspresi berdasarkan threshold
        expression = "Netral"
        
        # Terkejut (mata terbuka lebar dan mulut terbuka)
        if avg_ear > 0.25 and mar > 0.5 and eyebrow_pos < -0.02:
            expression = "Terkejut"
        # Senang (tersenyum - mulut terbuka sedang)
        elif mar > 0.3 and mar < 0.5 and avg_ear > 0.2:
            expression = "Senang"
        # Sedih (mata agak tertutup, mulut tertutup)
        elif avg_ear < 0.2 and mar < 0.2:
            expression = "Sedih"
        # Marah (alis turun, mata normal)
        elif eyebrow_pos > -0.01 and avg_ear > 0.18 and mar < 0.3:
            expression = "Marah"
        # Mulut terbuka (menguap atau berbicara)
        elif mar > 0.5:
            expression = "Mulut Terbuka"
        
        metrics = {
            'EAR': avg_ear,
            'MAR': mar,
            'Eyebrow': eyebrow_pos
        }
        
        return expression, metrics
    
    def draw_landmarks(self, image, landmarks, h, w):
        """Menggambar landmark wajah pada gambar"""
        # Gambar kontur mata kiri
        for i in self.LEFT_EYE:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Gambar kontur mata kanan
        for i in self.RIGHT_EYE:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Gambar kontur alis kiri
        for i in self.LEFT_EYEBROW:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        
        # Gambar kontur alis kanan
        for i in self.RIGHT_EYEBROW:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        
        # Gambar kontur mulut
        for i in self.MOUTH_OUTER:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    
    def process_frame(self, frame):
        """Memproses frame untuk deteksi wajah dan ekspresi"""
        # Konversi BGR ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        h, w, _ = frame.shape
        expression = "Tidak ada wajah"
        metrics = {}
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Gambar mesh wajah lengkap (opsional)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Gambar kontur penting
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Deteksi ekspresi
                expression, metrics = self.detect_expression(face_landmarks.landmark)
                
                # Gambar landmark khusus
                self.draw_landmarks(frame, face_landmarks.landmark, h, w)
        
        return frame, expression, metrics
    
    def run(self, source=0):
        """Menjalankan deteksi ekspresi wajah dari webcam atau video"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera/video")
            return
        
        print("Tekan 'q' untuk keluar")
        
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame")
                break
            
            # Flip frame untuk efek mirror
            frame = cv2.flip(frame, 1)
            
            # Proses frame
            processed_frame, expression, metrics = self.process_frame(frame)
            
            # Hitung FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # Tampilkan informasi
            cv2.putText(processed_frame, f"Ekspresi: {expression}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if metrics:
                y_offset = 70
                for key, value in metrics.items():
                    cv2.putText(processed_frame, f"{key}: {value:.3f}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30
            
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                       (10, processed_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Tampilkan frame
            cv2.imshow('Face Expression Detection - MediaPipe', processed_frame)
            
            # Keluar jika 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


if __name__ == "__main__":
    print("=== Deteksi Kontur Wajah dan Ekspresi dengan MediaPipe ===")
    print("Program ini akan mendeteksi wajah dan menganalisis ekspresi wajah.")
    print("\nEkspresi yang dapat dideteksi:")
    print("- Netral")
    print("- Senang")
    print("- Sedih")
    print("- Marah")
    print("- Terkejut")
    print("- Mulut Terbuka")
    print("\nMemulai kamera...")
    
    detector = FaceExpressionDetector()
    detector.run()
