# Dosya Bilgileri ve Kullanılan Teknolojiler

Bu dokümanda, projede oluşturulan tüm dosyaların amacı, işlevi ve kullanılan teknolojiler açıklanmaktadır.

---

## 📁 Python Script Dosyaları (.py)

### 1. `data_loader_pytorch.py`
**Amaç:** Veri yükleme ve ön işleme modülü  
**İşlevi:**
- ZIP dosyasından veri çıkarma
- Görüntüleri yükleme ve işleme
- Train-validation split yapma
- PyTorch Dataset ve DataLoader oluşturma
- Data augmentation uygulama

**Kullanılan Teknolojiler:**
- **PyTorch:** `torch.utils.data.Dataset`, `DataLoader`
- **Torchvision:** `transforms` (görüntü dönüşümleri)
- **PIL (Pillow):** Görüntü işleme
- **NumPy:** Array işlemleri
- **scikit-learn:** `LabelEncoder`, `train_test_split`
- **zipfile:** ZIP dosyası işleme

---

### 2. `model_pytorch.py`
**Amaç:** Model mimarisi tanımları  
**İşlevi:**
- CNN modeli oluşturma
- Transfer Learning modeli oluşturma (ResNet18)
- Model özeti gösterme

**Kullanılan Teknolojiler:**
- **PyTorch:** `torch.nn.Module`, `nn.Sequential`, `nn.Conv2d`, `nn.BatchNorm2d`, vb.
- **Torchvision:** `models.resnet18` (pre-trained model)

---

### 3. `train_pytorch.py`
**Amaç:** Model eğitim scripti  
**İşlevi:**
- Model eğitimi yönetimi
- Epoch bazında eğitim ve validasyon
- En iyi modeli kaydetme
- Early stopping uygulama
- Learning rate scheduling
- Eğitim geçmişini kaydetme

**Kullanılan Teknolojiler:**
- **PyTorch:** Model eğitimi, `torch.optim.Adam`, `torch.nn.CrossEntropyLoss`
- **Torchvision:** Pre-trained modeller
- **JSON:** Eğitim geçmişi kaydetme
- **Pickle:** Sınıf isimlerini kaydetme
- **argparse:** Komut satırı argümanları

---

### 4. `evaluate_pytorch.py`
**Amaç:** Model değerlendirme scripti  
**İşlevi:**
- Eğitilmiş modeli yükleme
- Validasyon seti üzerinde değerlendirme
- Test seti üzerinde tahmin yapma
- Confusion matrix oluşturma
- Classification report üretme
- Sonuçları kaydetme

**Kullanılan Teknolojiler:**
- **PyTorch:** Model yükleme ve inference
- **scikit-learn:** `accuracy_score`, `classification_report`, `confusion_matrix`
- **matplotlib:** Görselleştirme
- **seaborn:** Görselleştirme
- **JSON:** Sonuçları kaydetme

---

### 5. `create_detailed_results.py`
**Amaç:** Detaylı sonuç analizi ve görselleştirme  
**İşlevi:**
- ROC eğrileri oluşturma
- Precision-Recall eğrileri oluşturma
- Detaylı confusion matrix görselleştirme
- Performans metrikleri tablosu oluşturma
- Tüm sonuçları CSV ve JSON formatında kaydetme

**Kullanılan Teknolojiler:**
- **scikit-learn:** `roc_curve`, `auc`, `precision_recall_curve`, `average_precision_score`, `label_binarize`
- **matplotlib:** Grafik oluşturma
- **seaborn:** Görselleştirme (heatmap)
- **pandas:** Veri tabloları oluşturma
- **NumPy:** Array işlemleri
- **JSON:** Sonuçları kaydetme

---

### 6. `check_training_status.py`
**Amaç:** Eğitim durumunu kontrol etme  
**İşlevi:**
- Model dosyasının varlığını kontrol etme
- Eğitim geçmişini kontrol etme
- Eğitim loglarını okuma
- Durum raporu oluşturma

**Kullanılan Teknolojiler:**
- **JSON:** Eğitim geçmişi okuma
- **Pickle:** Sınıf isimlerini okuma
- **os:** Dosya sistemi işlemleri

---

### 7. `check_completion.py`
**Amaç:** Eğitimin tamamlanıp tamamlanmadığını kontrol etme  
**İşlevi:**
- Tamamlanma dosyasını kontrol etme
- Model dosyasının güncellik zamanını kontrol etme
- Bildirim gösterme

**Kullanılan Teknolojiler:**
- **os:** Dosya sistemi işlemleri
- **datetime:** Zaman işlemleri

---

### 8. `run_pytorch.py`
**Amaç:** Tam pipeline scripti (eğitim + değerlendirme)  
**İşlevi:**
- Eğitim ve değerlendirmeyi sırayla çalıştırma
- Hata yönetimi
- Özet rapor oluşturma

**Kullanılan Teknolojiler:**
- **Python:** Modül import ve çağırma
- **traceback:** Hata izleme

---

### 9. `run_complete_pipeline.py`
**Amaç:** TensorFlow versiyonu için tam pipeline (kullanılmadı)  
**Not:** Bu dosya TensorFlow için hazırlanmıştı ancak Python 3.14 uyumluluğu nedeniyle PyTorch kullanıldı.

---

### 10. `data_loader.py`
**Amaç:** TensorFlow versiyonu için veri yükleme (kullanılmadı)  
**Not:** TensorFlow için hazırlanmıştı, PyTorch versiyonu kullanıldı.

---

### 11. `model.py`
**Amaç:** TensorFlow versiyonu için model mimarisi (kullanılmadı)  
**Not:** TensorFlow/Keras için hazırlanmıştı, PyTorch versiyonu kullanıldı.

---

### 12. `train.py`
**Amaç:** TensorFlow versiyonu için eğitim scripti (kullanılmadı)  
**Not:** TensorFlow/Keras için hazırlanmıştı, PyTorch versiyonu kullanıldı.

---

### 13. `evaluate.py`
**Amaç:** TensorFlow versiyonu için değerlendirme scripti (kullanılmadı)  
**Not:** TensorFlow/Keras için hazırlanmıştı, PyTorch versiyonu kullanıldı.

---

## 📄 Markdown Dokümantasyon Dosyaları (.md)

### 1. `README.md`
**Amaç:** Ana proje dokümantasyonu  
**İçerik:**
- Proje özeti ve amacı
- Veri seti bilgileri
- Model mimarisi açıklaması
- Kurulum talimatları
- Kullanım kılavuzu
- Test sonuçları ve performans metrikleri
- Detaylı değerlendirme ve analiz
- Görselleştirmeler açıklaması
- Teknik detaylar

**Kullanılan Teknoloji:**
- **Markdown:** Dokümantasyon formatı

---

### 2. `DETAYLI_SONUCLAR.md`
**Amaç:** Kapsamlı sonuç analizi ve değerlendirme raporu  
**İçerik:**
- Genel performans metrikleri
- Sınıf bazında detaylı performans tabloları
- Confusion Matrix analizi
- ROC eğrileri analizi
- Precision-Recall eğrileri
- Detaylı metrik tabloları
- Model karşılaştırması
- Güçlü yönler ve iyileştirme alanları
- Hata analizi
- Sonuç ve öneriler

**Kullanılan Teknoloji:**
- **Markdown:** Dokümantasyon formatı

---

### 3. `SONUCLAR.md`
**Amaç:** Özet test sonuçları raporu  
**İçerik:**
- Eğitim bilgileri
- Test sonuçları özeti
- Model performansı
- Oluşturulan dosyalar listesi

**Kullanılan Teknoloji:**
- **Markdown:** Dokümantasyon formatı

---

### 4. `dosyabilgileri.md` (Bu dosya)
**Amaç:** Tüm dosyaların açıklaması ve teknoloji bilgileri

---

## 📊 Veri ve Sonuç Dosyaları

### 1. `performance_metrics.csv`
**Amaç:** Performans metriklerini CSV formatında saklama  
**İçerik:**
- Parazit türleri
- Precision, Recall, F1-Score, ROC AUC değerleri
- Support (örnek sayıları)
- Macro ve Weighted average değerleri

**Kullanılan Teknoloji:**
- **CSV:** Tablo veri formatı
- **pandas:** CSV dosyası oluşturma

---

### 2. `detailed_results.json`
**Amaç:** Detaylı sonuçları JSON formatında saklama  
**İçerik:**
- Accuracy değeri
- ROC AUC değerleri (micro ve macro)
- Confusion matrix
- Sınıf isimleri
- Performans metrikleri

**Kullanılan Teknoloji:**
- **JSON:** Yapılandırılmış veri formatı
- **Python json modülü:** JSON dosyası oluşturma

---

### 3. `test_predictions_pytorch.json`
**Amaç:** Test seti tahminlerini saklama  
**İçerik:**
- Tahmin edilen sınıflar
- Olasılık değerleri
- Sınıf isimleri

**Kullanılan Teknoloji:**
- **JSON:** Yapılandırılmış veri formatı

---

### 4. `requirements.txt`
**Amaç:** Gerekli Python kütüphanelerini listeleme  
**İçerik:**
- TensorFlow (kullanılmadı, Python 3.14 uyumsuzluğu nedeniyle)
- NumPy, Pillow, scikit-learn, matplotlib, seaborn, pandas

**Kullanılan Teknoloji:**
- **pip:** Python paket yöneticisi

---

## 🖼️ Görselleştirme Dosyaları (.png)

### 1. `confusion_matrix_detailed.png`
**Amaç:** Detaylı confusion matrix görselleştirmesi  
**Oluşturulma:** `create_detailed_results.py` scripti ile  
**Kullanılan Teknolojiler:**
- **matplotlib:** Grafik oluşturma
- **seaborn:** Heatmap görselleştirme
- **NumPy:** Array işlemleri

---

### 2. `confusion_matrix_normalized.png`
**Amaç:** Normalize edilmiş confusion matrix görselleştirmesi  
**Oluşturulma:** `create_detailed_results.py` scripti ile  
**Kullanılan Teknolojiler:**
- **matplotlib:** Grafik oluşturma
- **seaborn:** Heatmap görselleştirme (yüzde formatında)

---

### 3. `confusion_matrix_pytorch.png`
**Amaç:** Basit confusion matrix görselleştirmesi  
**Oluşturulma:** `evaluate_pytorch.py` scripti ile  
**Kullanılan Teknolojiler:**
- **matplotlib:** Grafik oluşturma
- **seaborn:** Heatmap görselleştirme

---

### 4. `roc_curves.png`
**Amaç:** ROC eğrileri görselleştirmesi  
**Oluşturulma:** `create_detailed_results.py` scripti ile  
**İçerik:**
- Her sınıf için ayrı ROC eğrisi
- Micro-average ROC eğrisi
- Macro-average ROC eğrisi
- Rastgele sınıflandırma referans çizgisi

**Kullanılan Teknolojiler:**
- **matplotlib:** Grafik oluşturma
- **scikit-learn:** `roc_curve`, `auc` hesaplama

---

### 5. `precision_recall_curves.png`
**Amaç:** Precision-Recall eğrileri görselleştirmesi  
**Oluşturulma:** `create_detailed_results.py` scripti ile  
**İçerik:**
- Her sınıf için Precision-Recall eğrisi
- Average Precision (AP) değerleri

**Kullanılan Teknolojiler:**
- **matplotlib:** Grafik oluşturma
- **scikit-learn:** `precision_recall_curve`, `average_precision_score`

---

### 6. `performance_table.png`
**Amaç:** Performans metrikleri tablosu görselleştirmesi  
**Oluşturulma:** `create_detailed_results.py` scripti ile  
**İçerik:**
- Tüm sınıflar için metrikler
- Renkli ve okunabilir tablo formatı

**Kullanılan Teknolojiler:**
- **matplotlib:** Tablo oluşturma
- **pandas:** Veri tablosu hazırlama

---

### 7. `training_history_pytorch.png`
**Amaç:** Eğitim geçmişi grafikleri  
**Oluşturulma:** `evaluate_pytorch.py` scripti ile  
**İçerik:**
- Training ve Validation Accuracy grafikleri
- Training ve Validation Loss grafikleri

**Kullanılan Teknolojiler:**
- **matplotlib:** Grafik oluşturma
- **JSON:** Eğitim geçmişi okuma

---

## 🔧 Yardımcı Script Dosyaları

### 1. `run.bat`
**Amaç:** Windows'ta kolay başlatma  
**İşlevi:**
- Python yolunu ayarlama
- Eğitimi başlatma
- Tamamlanma kontrolü
- Sonuçları gösterme

**Kullanılan Teknoloji:**
- **Batch Script:** Windows komut dosyası

---

### 2. `start_training.bat`
**Amaç:** Eğitimi başlatma ve izleme  
**İşlevi:**
- Eski bildirim dosyasını silme
- Eğitimi başlatma
- Tamamlanma kontrolü

**Kullanılan Teknoloji:**
- **Batch Script:** Windows komut dosyası

---

### 3. `monitor_training.ps1`
**Amaç:** Eğitim sürecini izleme  
**İşlevi:**
- Python sürecini kontrol etme
- Eğitim durumunu gösterme
- Tamamlanma bildirimi

**Kullanılan Teknoloji:**
- **PowerShell:** Windows otomasyon scripti

---

## 💾 Model ve Veri Dosyaları

### 1. `models/parasite_model_pytorch.pth`
**Amaç:** Eğitilmiş model dosyası  
**İçerik:**
- Model ağırlıkları (state_dict)
- Sınıf isimleri
- Sınıf sayısı

**Kullanılan Teknoloji:**
- **PyTorch:** Model kaydetme (`torch.save`)

---

### 2. `models/class_names_pytorch.pkl`
**Amaç:** Sınıf isimlerini saklama  
**İçerik:**
- Parazit türü isimleri listesi

**Kullanılan Teknoloji:**
- **Pickle:** Python nesne serileştirme

---

### 3. `models/training_history_pytorch.json`
**Amaç:** Eğitim geçmişini saklama  
**İçerik:**
- Epoch bazında training loss ve accuracy
- Epoch bazında validation loss ve accuracy

**Kullanılan Teknoloji:**
- **JSON:** Yapılandırılmış veri formatı

---

### 4. `EGITIM_TAMAMLANDI.txt`
**Amaç:** Eğitim tamamlanma bildirimi  
**İçerik:**
- Eğitim tamamlanma mesajı
- En iyi validation accuracy
- Tamamlanma zamanı

**Kullanılan Teknoloji:**
- **Text File:** Basit metin dosyası

---

### 5. `training_log.txt`
**Amaç:** Eğitim süreci log dosyası  
**İçerik:**
- Eğitim sırasında oluşan çıktılar
- Hata mesajları (varsa)
- İlerleme bilgileri

**Kullanılan Teknoloji:**
- **Text File:** Log dosyası
- **Tee-Object:** PowerShell'de çıktı yönlendirme

---

## 🗂️ Veri Klasörleri

### 1. `data/`
**Amaç:** Çıkarılan veri seti  
**İçerik:**
- `Chula-ParasiteEgg-11/` - Eğitim görüntüleri
- `_test/` - Test görüntüleri

**Kullanılan Teknoloji:**
- **zipfile:** ZIP dosyası çıkarma

---

### 2. `models/`
**Amaç:** Eğitilmiş modeller ve ilgili dosyalar  
**İçerik:**
- Model dosyası (.pth)
- Sınıf isimleri (.pkl)
- Eğitim geçmişi (.json)

---

## 📋 Özet: Kullanılan Ana Teknolojiler

### Derin Öğrenme Framework
- **PyTorch 2.9.1:** Model mimarisi, eğitim, inference
- **Torchvision 0.24.1:** Pre-trained modeller, görüntü dönüşümleri

### Veri İşleme
- **NumPy 2.3.3:** Array işlemleri
- **Pillow (PIL) 11.3.0:** Görüntü işleme
- **pandas 2.3.3:** Veri tabloları

### Makine Öğrenmesi
- **scikit-learn 1.7.2:** Metrik hesaplama, veri bölme, encoding

### Görselleştirme
- **matplotlib 3.10.7:** Grafik ve görselleştirme
- **seaborn 0.13.2:** İstatistiksel görselleştirme

### Veri Formatları
- **JSON:** Yapılandırılmış veri saklama
- **CSV:** Tablo veri formatı
- **Pickle:** Python nesne serileştirme

### Sistem ve Otomasyon
- **Python 3.14.1:** Ana programlama dili
- **PowerShell:** Windows otomasyon
- **Batch Script:** Windows komut dosyaları

### Dokümantasyon
- **Markdown:** Dokümantasyon formatı

---

## 🎯 Dosya Kullanım Senaryoları

### Eğitim Senaryosu
1. `data_loader_pytorch.py` - Verileri yükle
2. `model_pytorch.py` - Modeli oluştur
3. `train_pytorch.py` - Modeli eğit
4. `models/parasite_model_pytorch.pth` - Modeli kaydet

### Değerlendirme Senaryosu
1. `evaluate_pytorch.py` - Modeli değerlendir
2. `create_detailed_results.py` - Detaylı analiz yap
3. Görselleştirmeler oluşturulur (.png dosyaları)
4. Sonuçlar kaydedilir (.csv, .json dosyaları)

### Dokümantasyon Senaryosu
1. `README.md` - Genel proje bilgileri
2. `DETAYLI_SONUCLAR.md` - Kapsamlı sonuç analizi
3. `SONUCLAR.md` - Özet sonuçlar
4. `dosyabilgileri.md` - Bu dosya (dosya açıklamaları)

---

**Son Güncelleme:** 5 Aralık 2025  
**Toplam Dosya Sayısı:** 30+ dosya  
**Ana Teknoloji:** PyTorch, Transfer Learning (ResNet18)

