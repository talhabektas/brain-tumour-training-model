
import os
import argparse
import time

def run_all():
    """
    Tüm işlemleri sırayla çalıştırma
    """
    start_time = time.time()
    
    print("\n" + "="*50)
    print("BEYIN TÜMÖRÜ SINIFLANDIRMA PROJESİ")
    print("="*50 + "\n")
    
    # Veri setini indirme
    print("\n1. Veri Seti İndiriliyor...\n")
    os.system("python download_data.py")
    
    # Veri keşfi
    print("\n2. Veri Seti Keşfediliyor...\n")
    os.system("python data_exploration.py")
    
    # Veri ön işleme
    print("\n3. Veri Ön İşleme Kontrol Ediliyor...\n")
    os.system("python preprocessing.py")
    
    # Custom CNN modelini eğitme
    print("\n4. Custom CNN Modeli Eğitiliyor...\n")
    os.system("python custom_cnn.py")
    
    # Transfer learning modelini eğitme
    print("\n5. Transfer Learning Modeli Eğitiliyor...\n")
    os.system("python transfer_learning.py")
    
    # Model değerlendirmesi
    print("\n6. Modeller Değerlendiriliyor...\n")
    os.system("python model_evaluation.py")
    
    # Tamamlama süresi
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"TÜM İŞLEMLER TAMAMLANDI: {int(hours)}s {int(minutes)}d {int(seconds)}s")
    print("="*50 + "\n")

def run_specific_steps(steps):
    """
    Belirli adımları çalıştırma
    """
    step_scripts = {
        "download": "download_data.py",
        "explore": "data_exploration.py",
        "preprocess": "preprocessing.py",
        "custom_cnn": "custom_cnn.py",
        "transfer": "transfer_learning.py",
        "evaluate": "model_evaluation.py"
    }
    
    for step in steps:
        if step in step_scripts:
            print(f"\nÇalıştırılıyor: {step_scripts[step]}...\n")
            os.system(f"python {step_scripts[step]}")
        else:
            print(f"Bilinmeyen adım: {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Beyin Tümörü Sınıflandırma Projesi')
    parser.add_argument('--steps', nargs='+', choices=['download', 'explore', 'preprocess', 'custom_cnn', 'transfer', 'evaluate'],
                        help='Çalıştırılacak belirli adımlar')
    
    args = parser.parse_args()
    
    if args.steps:
        run_specific_steps(args.steps)
    else:
        run_all()