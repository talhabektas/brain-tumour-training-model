import os

def verify_dataset():
    # Veri seti klasörlerini kontrol et
    base_dir = 'data'
    training_dir = os.path.join(base_dir, 'Training')
    testing_dir = os.path.join(base_dir, 'Testing')
    
    if os.path.exists(training_dir) and os.path.exists(testing_dir):
        print("Veri seti klasörleri bulundu!")
        
        # Sınıf klasörlerini listele
        train_classes = os.listdir(training_dir)
        test_classes = os.listdir(testing_dir)
        
        print(f"Eğitim sınıfları: {train_classes}")
        print(f"Test sınıfları: {test_classes}")
        
        # Görüntü sayılarını say
        for cls in train_classes:
            train_count = len(os.listdir(os.path.join(training_dir, cls)))
            print(f"Eğitim sınıfı {cls}: {train_count} görüntü")
        
        for cls in test_classes:
            test_count = len(os.listdir(os.path.join(testing_dir, cls)))
            print(f"Test sınıfı {cls}: {test_count} görüntü")
            
        print("Veri seti hazır! Bir sonraki adıma geçebilirsiniz.")
    else:
        print("Veri seti klasörleri bulunamadı.")
        print("Lütfen veri setinin aşağıdaki konumda olduğundan emin olun:")
        print(f"{os.path.abspath(base_dir)}")

if __name__ == "__main__":
    verify_dataset()