import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def explore_dataset():
    # Veri klasör yapısını belirleme
    base_dir = 'data'  # Ya da 'BRAIN_TUMOR_CLASSIFICATION/data' şeklinde tam yol
    training_dir = os.path.join(base_dir, 'Training')
    testing_dir = os.path.join(base_dir, 'Testing')
    
    # Klasör yapısını kontrol etme
    if not os.path.exists(training_dir) or not os.path.exists(testing_dir):
        print("Veri seti klasörleri bulunamadı. Önce 'download_data.py' dosyasını çalıştırın.")
        return
    
    # Sınıfları listeleme
    classes = os.listdir(training_dir)
    print("Sınıflar:", classes)
    
    # Her sınıf için görüntü sayısı
    train_counts = {}
    test_counts = {}
    image_sizes = []
    
    for cls in classes:
        train_path = os.path.join(training_dir, cls)
        test_path = os.path.join(testing_dir, cls)
        
        train_images = os.listdir(train_path)
        test_images = os.listdir(test_path)
        
        train_counts[cls] = len(train_images)
        test_counts[cls] = len(test_images)
        
        # Örnek bir görüntünün boyutunu kontrol etme
        if train_images:
            sample_img_path = os.path.join(train_path, train_images[0])
            img = cv2.imread(sample_img_path)
            image_sizes.append(img.shape)
    
    print("\nEğitim Veri Seti Dağılımı:")
    for cls, count in train_counts.items():
        print(f"{cls}: {count} görüntü")
    
    print("\nTest Veri Seti Dağılımı:")
    for cls, count in test_counts.items():
        print(f"{cls}: {count} görüntü")
    
    print("\nGörüntü Boyutları (Örneklem):")
    for size in image_sizes:
        print(f"Boyut: {size} (Yükseklik, Genişlik, Kanal)")
    
    # Her sınıftan örnek görüntüler gösterme
    plt.figure(figsize=(16, 12))
    for i, cls in enumerate(classes):
        class_path = os.path.join(training_dir, cls)
        images = os.listdir(class_path)[:3]  # Her sınıftan 3 görüntü
        
        for j, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(4, 3, i*3 + j + 1)
            plt.imshow(img)
            plt.title(f"{cls} - Örnek {j+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data_samples.png')
    plt.show()
    
    print("\nÖrnek görüntüler 'data_samples.png' dosyasına kaydedildi.")

if __name__ == "__main__":
    explore_dataset()