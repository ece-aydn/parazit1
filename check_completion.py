"""
Eğitimin tamamlanıp tamamlanmadığını kontrol eder ve bildirim gösterir
"""

import os
import time
from datetime import datetime

def check_completion():
    """Eğitimin tamamlanıp tamamlanmadığını kontrol eder"""
    completion_file = "EGITIM_TAMAMLANDI.txt"
    model_file = "models/parasite_model_pytorch.pth"
    
    if os.path.exists(completion_file):
        print("\n" + "=" * 70)
        print("EGITIM TAMAMLANDI!")
        print("=" * 70)
        with open(completion_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print("=" * 70)
        return True
    elif os.path.exists(model_file):
        # Model var ama tamamlanma dosyası yok - hala çalışıyor olabilir
        model_time = datetime.fromtimestamp(os.path.getmtime(model_file))
        time_diff = (datetime.now() - model_time).total_seconds()
        
        if time_diff < 300:  # 5 dakikadan az
            print(f"Egitim devam ediyor... (Son model guncellemesi: {int(time_diff)} saniye once)")
        else:
            print("Model dosyasi var ama egitim tamamlanma dosyasi yok.")
            print("Egitim yarida kalmis olabilir.")
        return False
    else:
        print("Egitim henuz baslamadi veya model dosyasi olusturulmadi.")
        return False

if __name__ == '__main__':
    check_completion()






