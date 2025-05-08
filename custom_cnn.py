import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from preprocessing import create_data_generators

def create_custom_cnn(input_shape=(224, 224, 3), num_classes=4):
    """
    Özel CNN modeli oluşturma
    """
    model = models.Sequential([
        # Giriş katmanı
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # İkinci evrişim bloğu
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Üçüncü evrişim bloğu
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dördüncü evrişim bloğu
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Tam bağlantı katmanları
        layers.Flatten(),
        layers.Dropout(0.5),  # Aşırı öğrenmeyi (overfitting) önlemek için
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Çıkış katmanı
    ])
    
    # Model derleme
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_custom_cnn(epochs=50, batch_size=32, img_width=224, img_height=224):
    """
    Custom CNN modelini eğitme
    """
    # Model klasörünü oluşturma
    model_dir = 'models'
    logs_dir = 'logs/custom_cnn'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Veri yükleyicilerini oluşturma
    train_generator, test_generator, class_indices = create_data_generators(
        img_width=img_width, 
        img_height=img_height,
        batch_size=batch_size
    )
    
    if not train_generator or not test_generator:
        return None, None
    
    # Custom CNN modelini oluşturma
    model = create_custom_cnn(input_shape=(img_width, img_height, 3), num_classes=len(class_indices))
    model.summary()
    
    # Callback'leri oluşturma
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_custom_cnn_model.h5'),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1
    )
    
    # Modeli eğitme
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback]
    )
    
    # Son modeli kaydetme
    model.save(os.path.join(model_dir, 'final_custom_cnn_model.h5'))
    
    print(f"Model kaydedildi: {os.path.join(model_dir, 'final_custom_cnn_model.h5')}")
    
    # Eğitim geçmişini görselleştirme
    plot_training_history(history, 'Custom CNN', save_path='plots/custom_cnn_history.png')
    
    return model, history

def plot_training_history(history, title, save_path=None):
    """
    Eğitim geçmişini görselleştirme
    """
    # Plots klasörünü kontrol etme ve oluşturma
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Eğitim Doğruluğu')
    plt.plot(epochs, val_acc, 'r-', label='Doğrulama Doğruluğu')
    plt.title(f'{title} - Doğruluk')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Eğitim Kaybı')
    plt.plot(epochs, val_loss, 'r-', label='Doğrulama Kaybı')
    plt.title(f'{title} - Kayıp')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Eğitim geçmişi grafiği kaydedildi: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Custom CNN modelini eğitme
    model, history = train_custom_cnn(epochs=50, batch_size=32)