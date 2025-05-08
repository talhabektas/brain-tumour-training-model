import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(img_width=224, img_height=224, batch_size=32):
    base_dir = 'data'
    training_dir = os.path.join(base_dir, 'Training')
    testing_dir = os.path.join(base_dir, 'Testing')
    
    if not os.path.exists(training_dir) or not os.path.exists(testing_dir):
        print("Veri seti klasörleri bulunamadı. Önce 'download_data.py' dosyasını çalıştırın.")
        return None, None, None
    
    # Veri artırma (data augmentation) - eğitim seti için
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Test seti için sadece yeniden ölçeklendirme
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Eğitim ve test verilerini yükleme
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        testing_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Sınıf indekslerini alma
    class_indices = train_generator.class_indices
    print("Sınıf indeksleri:", class_indices)
    
    return train_generator, test_generator, class_indices

if __name__ == "__main__":
    # Test amaçlı çalıştırma
    train_gen, test_gen, class_indices = create_data_generators()
    
    if train_gen and test_gen:
        print(f"Eğitim örnekleri: {train_gen.samples}")
        print(f"Test örnekleri: {test_gen.samples}")
        print(f"Sınıf sayısı: {len(class_indices)}")
        
        # Örnek bir batch alıp boyutlarını kontrol etme
        X_batch, y_batch = next(train_gen)
        print(f"Batch boyutu: {X_batch.shape}, Etiket boyutu: {y_batch.shape}")