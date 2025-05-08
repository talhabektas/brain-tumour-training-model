import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from preprocessing import create_data_generators

def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=4):
    """
    VGG16 tabanlı transfer learning modeli oluşturma
    """
    # VGG16 temel modelini yükleme (ImageNet ağırlıkları ile)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Temel modelin katmanlarını dondurma
    for layer in base_model.layers:
        layer.trainable = False
    
    # Yeni katmanlar ekleme
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Çıkış katmanı
    ])
    
    # Model derleme
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_transfer_learning_model(epochs_initial=20, epochs_fine_tuning=10, batch_size=32, img_width=224, img_height=224):
    """
    Transfer learning modelini eğitme
    """
    # Model klasörünü oluşturma
    model_dir = 'models'
    logs_dir = 'logs/transfer_learning'
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
        return None, None, None
    
    # Transfer learning modelini oluşturma
    model = create_transfer_learning_model(input_shape=(img_width, img_height, 3), num_classes=len(class_indices))
    model.summary()
    
    # Callback'leri oluşturma
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_transfer_model.h5'),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(logs_dir, 'initial'),
        histogram_freq=1
    )
    
    # İlk eğitim (donmuş katmanlarla)
    print("Transfer learning modeli eğitiliyor (ilk aşama)...")
    history_initial = model.fit(
        train_generator,
        epochs=epochs_initial,
        validation_data=test_generator,
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback]
    )
    
    # Fine-tuning - Son birkaç VGG16 katmanını eğitilebilir yapma
    print("Fine-tuning başlatılıyor...")
    # VGG16'nın son 4 katmanını eğitilebilir yapma
    for layer in model.layers[0].layers[-4:]:
        layer.trainable = True
    
    # Daha düşük öğrenme oranıyla yeniden derleme
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model_checkpoint_fine_tuning = ModelCheckpoint(
        os.path.join(model_dir, 'best_fine_tuned_model.h5'),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    tensorboard_callback_fine_tuning = TensorBoard(
        log_dir=os.path.join(logs_dir, 'fine_tuning'),
        histogram_freq=1
    )
    
    # Fine-tuning eğitimi
    history_fine_tuning = model.fit(
        train_generator,
        epochs=epochs_fine_tuning,
        validation_data=test_generator,
        callbacks=[early_stopping, model_checkpoint_fine_tuning, tensorboard_callback_fine_tuning]
    )
    
    # Son modeli kaydetme
    model.save(os.path.join(model_dir, 'final_transfer_learning_model.h5'))
    
    print(f"Model kaydedildi: {os.path.join(model_dir, 'final_transfer_learning_model.h5')}")
    
    # Eğitim geçmişini görselleştirme
    plot_training_history(history_initial, 'Transfer Learning (İlk Aşama)', save_path='plots/transfer_initial_history.png')
    plot_training_history(history_fine_tuning, 'Transfer Learning (Fine-Tuning)', save_path='plots/transfer_fine_tuning_history.png')
    
    return model, history_initial, history_fine_tuning

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
    # Transfer learning modelini eğitme
    model, history_initial, history_fine_tuning = train_transfer_learning_model(
        epochs_initial=20, 
        epochs_fine_tuning=10,
        batch_size=32
    )