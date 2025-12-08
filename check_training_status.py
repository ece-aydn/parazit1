"""
Eğitim durumunu kontrol eden script
"""

import os
import json
from datetime import datetime

def check_training_status():
    """Eğitim durumunu kontrol eder"""
    print("=" * 60)
    print("EĞİTİM DURUMU KONTROLÜ")
    print("=" * 60)
    print()
    
    # 1. Model dosyası kontrolü
    model_path = "models/parasite_model_pytorch.pth"
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        model_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"[OK] Model dosyası bulundu!")
        print(f"  - Konum: {model_path}")
        print(f"  - Boyut: {model_size:.2f} MB")
        print(f"  - Son güncelleme: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        model_exists = True
    else:
        print("[X] Model dosyası henuz olusturulmadi")
        print(f"  - Beklenen konum: {model_path}")
        print()
        model_exists = False
    
    # 2. Eğitim geçmişi kontrolü
    history_path = "models/training_history_pytorch.json"
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if history and 'train_acc' in history:
            num_epochs = len(history['train_acc'])
            final_train_acc = history['train_acc'][-1]
            final_val_acc = history['val_acc'][-1]
            best_val_acc = max(history['val_acc'])
            
            print(f"[OK] Egitim gecmisi bulundu!")
            print(f"  - Tamamlanan epoch sayısı: {num_epochs}")
            print(f"  - Son eğitim doğruluğu: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
            print(f"  - Son validasyon doğruluğu: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
            print(f"  - En iyi validasyon doğruluğu: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
            print()
            history_exists = True
        else:
            print("[X] Egitim gecmisi bos veya eksik")
            print()
            history_exists = False
    else:
        print("[X] Egitim gecmisi dosyasi bulunamadi")
        print(f"  - Beklenen konum: {history_path}")
        print()
        history_exists = False
    
    # 3. Sınıf isimleri kontrolü
    class_names_path = "models/class_names_pytorch.pkl"
    if os.path.exists(class_names_path):
        print(f"[OK] Sinif isimleri dosyasi bulundu: {class_names_path}")
        print()
        class_names_exists = True
    else:
        print(f"[X] Sinif isimleri dosyasi bulunamadi: {class_names_path}")
        print()
        class_names_exists = False
    
    # 4. Log dosyası kontrolü
    log_path = "training_log.txt"
    if os.path.exists(log_path):
        log_size = os.path.getsize(log_path) / 1024  # KB
        log_time = datetime.fromtimestamp(os.path.getmtime(log_path))
        print(f"[OK] Egitim log dosyasi bulundu!")
        print(f"  - Konum: {log_path}")
        print(f"  - Boyut: {log_size:.2f} KB")
        print(f"  - Son güncelleme: {log_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Son satırları göster
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    print("Son 10 satır:")
                    for line in lines[-10:]:
                        print(f"  {line.rstrip()}")
        except:
            pass
        print()
    else:
        print("[X] Egitim log dosyasi bulunamadi")
        print()
    
    # Özet
    print("=" * 60)
    print("ÖZET")
    print("=" * 60)
    
    if model_exists and history_exists:
        print("[OK] [OK] [OK] EGITIM TAMAMLANDI! [OK] [OK] [OK]")
        print()
        print("Şimdi test edebilirsiniz:")
        print("  python evaluate_pytorch.py")
        print()
        print("Veya tam pipeline:")
        print("  python run_pytorch.py")
    elif history_exists:
        print("[!] Egitim devam ediyor... (Gecmis var ama model henuz kaydedilmedi)")
    else:
        print("[...] Egitim henuz baslamadi veya cok erken asamada...")
    
    print("=" * 60)

if __name__ == '__main__':
    check_training_status()

