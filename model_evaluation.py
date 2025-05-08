import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import random
from preprocessing import create_data_generators

def load_trained_models():
    """
    Eğitilmiş modelleri yükleme
    """
    models_dir = 'models'
    
    # Model dosyalarını kontrol etme
    custom_model_path = os.path.join(models_dir, 'best_custom_cnn_model.h5')
    transfer_model_path = os.path.join(models_dir, 'best_fine_tuned_model.h5')
    
    custom_model = None
    transfer_model = None
    
    if os.path.exists(custom_model_path):
        print(f"Custom CNN modeli yükleniyor: {custom_model_path}")
        custom_model = load_model(custom_model_path)
    else:
        print(f"Custom CNN modeli bulunamadı: {custom_model_path}")
    
    if os.path.exists(transfer_model_path):
        print(f"Transfer Learning modeli yükleniyor: {transfer_model_path}")
        transfer_model = load_model(transfer_model_path)
    else:
        print(f"Transfer Learning modeli bulunamadı: {transfer_model_path}")
    
    return custom_model, transfer_model

def evaluate_model(model, generator, model_name):
    """
    Modeli test seti üzerinde değerlendirme
    """
    if model is None:
        print(f"{model_name} modeli değerlendirilemedi (model yok)")
        return None, None, None, None
    
    # Plots klasörünü kontrol etme ve oluşturma
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Model değerlendirmesi
    print(f"{model_name} değerlendiriliyor...")
    test_loss, test_accuracy = model.evaluate(generator)
    print(f"{model_name} Test Doğruluğu: {test_accuracy:.4f}")
    print(f"{model_name} Test Kaybı: {test_loss:.4f}")
    
    # Tahminler
    print("Tahminler yapılıyor...")
    predictions = model.predict(generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes
    
    # Sınıf isimleri
    class_names = list(generator.class_indices.keys())
    
    # Confusion matrix
    print("Confusion matrix oluşturuluyor...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.show()
    
    # Sınıflandırma raporu
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"{model_name} Sınıflandırma Raporu:")
    print(report)
    
    # Raporu dosyaya kaydetme
    with open(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_report.txt'), 'w') as f:
        f.write(f"{model_name} Sınıflandırma Raporu:\n")
        f.write(report)
    
    return test_accuracy, test_loss, cm, report

def predict_and_display_images(model, model_name, img_width=224, img_height=224):
    """
    Test görüntülerinde tahminler yapma ve görselleştirme
    """
    if model is None:
        print(f"{model_name} ile tahmin yapılamadı (model yok)")
        return
    
    # Plots klasörünü kontrol etme ve oluşturma
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Veri seti yollarını belirleme
    base_dir = 'data/brain_tumor_dataset'
    testing_dir = os.path.join(base_dir, 'Testing')
    
    # Klasör yapısını kontrol etme
    if not os.path.exists(testing_dir):
        print("Test veri seti klasörü bulunamadı.")
        return
    
    # Sınıfları listeleme
    classes = os.listdir(testing_dir)
    
    # Test klasöründen her sınıftan bir görüntü seç
    test_images = []
    true_labels = []
    
    for i, cls in enumerate(classes):
        class_path = os.path.join(testing_dir, cls)
        images = random.sample(os.listdir(class_path), 1)
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_images.append(img)
            true_labels.append(cls)
    
    # Görüntüleri modele uygun şekilde önişleme
    processed_images = []
    for img in test_images:
        resized_img = cv2.resize(img, (img_width, img_height))
        normalized_img = resized_img / 255.0
        processed_images.append(normalized_img)
    
    processed_images = np.array(processed_images)
    
    # Tahminler
    predictions = model.predict(processed_images)
    pred_classes = [classes[np.argmax(pred)] for pred in predictions]
    
    # Sonuçları görselleştirme
    plt.figure(figsize=(15, 10))
    for i, (img, true_label, pred_label) in enumerate(zip(test_images, true_labels, pred_classes)):
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label}", color=color)
        plt.axis('off')
    
    plt.suptitle(f"{model_name} - Test Görüntüleri Tahminleri", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_predictions.png'))
    plt.show()
    
    print(f"Tahmin görselleştirmeleri kaydedildi: {os.path.join(plots_dir, f'{model_name.lower().replace(' ', '_')}_predictions.png')}")

def compare_models(custom_acc, custom_loss, transfer_acc, transfer_loss):
    """
    Model performanslarını karşılaştırma ve görselleştirme
    """
    # Plots klasörünü kontrol etme ve oluşturma
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Karşılaştırma verilerini hazırlama
    model_names = ['Custom CNN', 'Transfer Learning (VGG16)']
    accuracies = [custom_acc, transfer_acc]
    losses = [custom_loss, transfer_loss]
    
    # Doğruluk grafiği
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, accuracies, color=['blue', 'green'])
    plt.title('Model Doğruluk Karşılaştırması')
    plt.ylabel('Doğruluk')
    plt.ylim(0, 1)
    
    # Değerleri çubukların üzerine yazma
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, losses, color=['blue', 'green'])
    plt.title('Model Kayıp Karşılaştırması')
    plt.ylabel('Kayıp')
    
    # Değerleri çubukların üzerine yazma
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
    plt.show()
    
    print(f"Model karşılaştırma grafiği kaydedildi: {os.path.join(plots_dir, 'model_comparison.png')}")
    
    # Karşılaştırma sonuçlarını dosyaya kaydetme
    with open(os.path.join(plots_dir, 'model_comparison.txt'), 'w') as f:
        f.write("Model Performans Karşılaştırması\n")
        f.write("===============================\n\n")
        f.write(f"Custom CNN Doğruluk: {custom_acc:.4f}\n")
        f.write(f"Custom CNN Kayıp: {custom_loss:.4f}\n\n")
        f.write(f"Transfer Learning Doğruluk: {transfer_acc:.4f}\n")
        f.write(f"Transfer Learning Kayıp: {transfer_loss:.4f}\n\n")
        f.write(f"Doğruluk Farkı: {abs(transfer_acc - custom_acc):.4f} (Transfer Learning {'daha iyi' if transfer_acc > custom_acc else 'daha kötü'})\n")
        f.write(f"Kayıp Farkı: {abs(transfer_loss - custom_loss):.4f} (Transfer Learning {'daha iyi' if transfer_loss < custom_loss else 'daha kötü'})\n")

def run_evaluation():
    """
    Tüm değerlendirme işlemlerini çalıştırma
    """
    print("Model değerlendirmesi başlatılıyor...")
    
    # Veri yükleyicilerini oluşturma
    _, test_generator, _ = create_data_generators()
    
    if not test_generator:
        print("Test veri yükleyicisi oluşturulamadı.")
        return
    
    # Eğitilmiş modelleri yükleme
    custom_model, transfer_model = load_trained_models()
    
    # Modelleri değerlendirme
    custom_acc, custom_loss, custom_cm, custom_report = None, None, None, None
    transfer_acc, transfer_loss, transfer_cm, transfer_report = None, None, None, None
    
    if custom_model:
        custom_acc, custom_loss, custom_cm, custom_report = evaluate_model(custom_model, test_generator, "Custom CNN")
        predict_and_display_images(custom_model, "Custom CNN")
    
    if transfer_model:
        transfer_acc, transfer_loss, transfer_cm, transfer_report = evaluate_model(transfer_model, test_generator, "Transfer Learning")
        predict_and_display_images(transfer_model, "Transfer Learning")
    
    # Model performanslarını karşılaştırma (her iki model de mevcutsa)
    if custom_model and transfer_model:
        compare_models(custom_acc, custom_loss, transfer_acc, transfer_loss)

if __name__ == "__main__":
    run_evaluation()

