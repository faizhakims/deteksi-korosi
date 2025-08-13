import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox, Button, Label, Frame, TOP, BOTTOM, BOTH
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning

class KorosiDetector:
    def __init__(self):
        self.keparahan_labels = ["Rendah", "Sedang", "Tinggi"]
        self.keparahan_colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]  # kuning, orange, merah
    
    def generate_gabor_filters(self, ksize=15, sigma=4.0):
        filters = []
        orientations = 8
        for theta in np.arange(0, np.pi, np.pi / orientations):
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kernel /= 1.5 * kernel.sum()
            filters.append(kernel)
        return filters
    
    def apply_gabor_filter(self, img, filters):
        gabor_responses = []
        for kernel in filters:
            response = cv2.filter2D(img, cv2.CV_32F, kernel)
            gabor_responses.append(response)
        
        gabor_response = np.max(gabor_responses, axis=0)
        return cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def compute_lbp(self, gray, radius=3, n_points=24):
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        return cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def manual_entropy(self, gray, kernel_size=9):
        gray_norm = gray.astype(np.float32) / 255.0
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        hist = cv2.filter2D(gray_norm, -1, kernel)
        eps = 1e-7  # log(0)
        entropy = -hist * np.log2(hist + eps)
        return cv2.normalize(entropy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def get_rust_color_mask(self, hsv):
        lower_dark_rust = np.array([0, 50, 30])
        upper_dark_rust = np.array([20, 255, 150])
        
        lower_light_rust = np.array([5, 50, 150])
        upper_light_rust = np.array([25, 255, 255])
        
        lower_red = np.array([160, 50, 30])
        upper_red = np.array([180, 255, 255])
        
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([30, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_dark_rust, upper_dark_rust)
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        mask3 = cv2.inRange(hsv, lower_light_rust, upper_light_rust)
        mask4 = cv2.inRange(hsv, lower_orange, upper_orange)
        return cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), cv2.bitwise_or(mask3, mask4))
    
    def enhance_texture_detection(self, gray):
        entropy = self.manual_entropy(gray)
        
        lbp = self.compute_lbp(gray)
        
        gabor_filters = self.generate_gabor_filters()
        gabor_response = self.apply_gabor_filter(gray, gabor_filters)
        
        texture_features = cv2.addWeighted(entropy, 0.4, lbp, 0.3, 0)
        texture_features = cv2.addWeighted(texture_features, 0.7, gabor_response, 0.3, 0)
        
        return texture_features
    
    def segment_corrosion_kmeans(self, img, mask):
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        pixels = masked_img.reshape(-1, 3)
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
        
        if pixels.size == 0:
            return np.zeros_like(mask)
        
        k = 3
        if len(pixels) < k:
            k = max(1, len(pixels))
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
        
        result_mask = np.zeros_like(mask)
        flat_mask = mask.flatten()
        non_zero_indices = np.where(flat_mask > 0)[0]
        
        if len(non_zero_indices) > 0:
            all_labels = np.zeros(flat_mask.shape, dtype=np.uint8)
            all_labels[non_zero_indices] = kmeans.labels_ + 1
            result_mask = all_labels.reshape(mask.shape)
        
        return result_mask
    
    def analyze_severity(self, img, severity_mask):
        severity_levels = [0, 0, 0]
        
        for i in range(1, 4):
            severity_levels[i-1] = np.sum(severity_mask == i)
        
        total_corrosion = sum(severity_levels)
        if total_corrosion == 0:
            return img, [0, 0, 0]
        severity_percent = [round(100 * level / total_corrosion, 1) for level in severity_levels]
        severity_viz = img.copy()
        for i in range(1, 4):
            severity_viz[severity_mask == i] = self.keparahan_colors[i-1]
        
        return severity_viz, severity_percent
    
    def deteksi_korosi(self, img):
        original = img.copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
        mask_warna = self.get_rust_color_mask(hsv)
        gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
        texture_features = self.enhance_texture_detection(gray)
        hist = cv2.calcHist([texture_features], [0], None, [256], [0, 256])
        threshold_value = np.argmax(hist[50:]) + 50
        _, mask_tekstur = cv2.threshold(texture_features, threshold_value, 255, cv2.THRESH_BINARY)
        mask_gabung = cv2.bitwise_and(mask_warna, mask_tekstur)
        kernel = np.ones((5, 5), np.uint8)
        mask_bersih = cv2.morphologyEx(mask_gabung, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_bersih = cv2.morphologyEx(mask_bersih, cv2.MORPH_OPEN, kernel, iterations=1)
        severity_mask = self.segment_corrosion_kmeans(img, mask_bersih)
        severity_viz, severity_percent = self.analyze_severity(img, severity_mask)
        overlay = img.copy()
        for i in range(1, 4):
            overlay[severity_mask == i] = self.keparahan_colors[i-1]
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        contours, _ = cv2.findContours(mask_bersih, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 50
        kontur_terfilter = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        kontur_terfilter = sorted(kontur_terfilter, key=cv2.contourArea, reverse=True)
        hasil = blended.copy()
        total_area = img.shape[0] * img.shape[1]
        corrosion_areas = []
        
        for i, cnt in enumerate(kontur_terfilter):
            area = cv2.contourArea(cnt)
            area_percent = round((area / total_area) * 100, 2)
            corrosion_areas.append(area_percent)
            
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(hasil, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if i < 5:
                label = f"{area_percent}%"
                cv2.putText(hasil, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        total_corrosion_percent = round(sum(corrosion_areas), 2)
        
        return {
            'original': original,
            'hasil': hasil, 
            'mask': mask_bersih,
            'overlay': overlay,
            'blended': blended,
            'severity_viz': severity_viz,
            'severity_percent': severity_percent,
            'total_corrosion_percent': total_corrosion_percent
        }

class KorosiDetectorGUI:
    def __init__(self):
        self.detector = KorosiDetector()
        self.current_image_path = None
        self.output_path = None
        self.results = None
        
    def select_image(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()
        
        if file_path:
            self.current_image_path = file_path
            return file_path
        return None
        
    def process_image(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Gagal membaca gambar")
            return None
            
        self.results = self.detector.deteksi_korosi(img)
        return self.results
        
    def create_visualization(self):
        if not self.results:
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(self.results['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Gambar Asli")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(self.results['blended'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Deteksi Awal")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(self.results['severity_viz'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Tingkat Keparahan")
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(self.results['mask'], cmap='gray')
        axes[1, 0].set_title("Mask Korosi")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(self.results['overlay'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Overlay Area Korosi")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(self.results['hasil'], cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f"Hasil Akhir (Total Korosi: {self.results['total_corrosion_percent']}%)")
        axes[1, 2].axis('off')
        
        keparahan = self.results['severity_percent']
        fig.text(0.5, 0.01, 
                f"Distribusi Keparahan Korosi: Rendah {keparahan[0]}%, Sedang {keparahan[1]}%, Tinggi {keparahan[2]}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        direktori, nama_file = os.path.split(self.current_image_path)
        nama, ekstensi = os.path.splitext(nama_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(direktori, f"{nama}_analisis_{timestamp}.png")
        
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        baris_atas = np.hstack((self.results['original'], self.results['blended'], self.results['severity_viz']))
        baris_bawah = np.hstack((cv2.cvtColor(self.results['mask'], cv2.COLOR_GRAY2BGR), 
                                self.results['overlay'], 
                                self.results['hasil']))
        
        if baris_atas.shape[1] != baris_bawah.shape[1]:
            baris_bawah = cv2.resize(baris_bawah, (baris_atas.shape[1], baris_bawah.shape[0]))
            
        gabungan = np.vstack((baris_atas, baris_bawah))
        
        cv2.putText(gabungan, f"Total Area Korosi: {self.results['total_corrosion_percent']}%", 
                    (10, gabungan.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
        kep_text = f"Distribusi: Rendah {keparahan[0]}%, Sedang {keparahan[1]}%, Tinggi {keparahan[2]}%"
        cv2.putText(gabungan, kep_text, (10, gabungan.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return gabungan
        
    def show_and_save(self, img_gabungan):
        cv2.namedWindow('Deteksi Korosi', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Deteksi Korosi', 1280, 720)
        cv2.imshow('Deteksi Korosi', img_gabungan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # direktori, nama_file = os.path.split(self.current_image_path)
        # nama, ekstensi = os.path.splitext(nama_file)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # path_hasil = os.path.join(direktori, f"{nama}_hasil_{timestamp}{ekstensi}")
        
        # if cv2.imwrite(path_hasil, img_gabungan):
        #     print(f"Hasil OpenCV disimpan di: {path_hasil}")
        #     print(f"Visualisasi matplotlib disimpan di: {self.output_path}")
        #     return True
        # else:
        #     print("Gagal menyimpan hasil")
        #     return False
    
    def run_app(self):
        root = Tk()
        root.title("Aplikasi Deteksi Korosi")
        root.geometry("400x200")
        
        # Frame
        frame = Frame(root)
        frame.pack(fill=BOTH, expand=True)
        
        # Label
        label = Label(frame, text="Aplikasi Deteksi Korosi yang Ditingkatkan", font=("Arial", 14))
        label.pack(side=TOP, pady=20)
        
        # Tombol pilih gambar
        btn_select = Button(frame, text="Pilih Gambar", font=("Arial", 12), 
                        command=self.process_selected_image)
        btn_select.pack(pady=20)
        
        # Tombol keluar
        btn_exit = Button(frame, text="Keluar", font=("Arial", 12), command=root.destroy)
        btn_exit.pack(side=BOTTOM, pady=20)

        root.mainloop()
    
    def process_selected_image(self):
        file_path = self.select_image()
        if not file_path:
            return
            
        results = self.process_image(file_path)
        if not results:
            return
            
        visualization = self.create_visualization()
        if visualization is None:
            messagebox.showerror("Error", "Gagal membuat visualisasi")
            return
            
        success = self.show_and_save(visualization)
        if success:
            messagebox.showinfo("Sukses", f"Analisis selesai!\nTotal area korosi: {self.results['total_corrosion_percent']}%\n"f"Hasil disimpan di folder gambar asli.")

def main():
    app = KorosiDetectorGUI()
    app.run_app()

if __name__ == "__main__":
    main()