import numpy as np
import os
from PIL import Image

class KNearestNeighbor:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X_test, metric='L2'):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        if metric == 'L1':
            for i in range(num_test):
                # Manhattan Mesafesi
                dists[i, :] = np.sum(np.abs(self.X_train - X_test[i, :]), axis=1)
        else:
            for i in range(num_test):
                # Öklid Mesafesi
                dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X_test[i, :]), axis=1))
        return dists

    def predict(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # En yakın k komşunun etiketlerini al
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            # En çok tekrar eden etiketi bul
            y_pred[i] = np.argmax(np.bincount(closest_y.astype(int)))
        return y_pred

def load_images_from_folder(folder_path, max_per_class=100):
    
    X, y = [], []
    
    # Sınıf klasörlerini listele (0, 1, 2... veya adlar)
    class_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    print(f"'{os.path.basename(folder_path)}' klasöründe {len(class_dirs)} sınıf bulundu.")

    for idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(folder_path, class_name)
        images = os.listdir(class_path)[:max_per_class] # Hız için her sınıftan kısıtlı sayıda al
        
        for img_name in images:
            try:
                img_path = os.path.join(class_path, img_name)
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((32, 32))
                    X.append(np.array(img).flatten())
                    # Eğer klasör adı sayıysa onu kullan, değilse indexi kullan
                    y.append(int(class_name) if class_name.isdigit() else idx)
            except:
                continue
                
    return np.array(X), np.array(y)

def main():
    # DOSYA YOLUN
    base_path = r"C:\Users\canbe\Downloads\cifar10\cifar10"
    
    print("--- CIFAR-10 k-NN Terminal Uygulaması ---")

    # 1. Veri Yükleme (Eğitim için her sınıftan 200, test için 20 resim alıyoruz - HIZ İÇİN)
    print("\nVeriler okunuyor, lütfen bekleyin...")
    X_train, y_train = load_images_from_folder(os.path.join(base_path, 'train'), max_per_class=200)
    X_test, y_test = load_images_from_folder(os.path.join(base_path, 'test'), max_per_class=20)

    if X_train.size == 0:
        print("Hata: Veri yüklenemedi! Lütfen dosya yolunu ve klasör yapısını kontrol edin.")
        return

    # 2. Kullanıcı Seçimleri
    print("\n[1] Manhattan (L1)")
    print("[2] Öklid (L2)")
    metric_choice = input("Mesafe ölçütü seçin (1/2): ")
    metric = 'L1' if metric_choice == '1' else 'L2'
    
    k_val = int(input("k değerini girin (Örn: 3): "))

    # 3. Algoritmayı Çalıştırma
    knn = KNearestNeighbor()
    knn.train(X_train, y_train)
    
    print(f"\n{metric} mesafesi kullanılarak {len(X_test)} test verisi sınıflandırılıyor...")
    dists = knn.compute_distances(X_test, metric=metric)
    y_pred = knn.predict(dists, k=k_val)

    # 4. Sonuç
    accuracy = np.mean(y_pred == y_test)
    print("-" * 30)
    print(f"BAŞARI ORANI: %{accuracy * 100:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    main()