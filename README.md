# Parazit Yumurta Tespiti - Mikroskop Görüntü İşleme Projesi

## 📋 Proje Özeti

Bu proje, mikroskop görüntülerinden parazit yumurtası tespiti yapmak için derin öğrenme (Deep Learning) teknikleri kullanmaktadır. Chula-ParasiteEgg-11 veri seti ile eğitilmiş bir Transfer Learning modeli (ResNet18) içerir. Model, %98.87 doğruluk oranı ile çok yüksek bir performans sergilemektedir.

**Proje Türü:** Tıbbi Görüntü İşleme - Sınıflandırma  
**Kullanılan Teknoloji:** PyTorch, Transfer Learning (ResNet18)  
**Veri Seti:** Chula-ParasiteEgg-11  

---

## 🎯 Proje Amacı

Mikroskop görüntülerinden parazit yumurtalarını otomatik olarak tespit etmek ve sınıflandırmak. Bu proje, tıbbi laboratuvarlarda parazit analiz sürecini hızlandırmak ve doğruluğu artırmak amacıyla geliştirilmiştir.

---

## 📊 Veri Seti Bilgileri

### Chula-ParasiteEgg-11 Veri Seti

- **Toplam Görüntü Sayısı:** 13,200
- **Eğitim Görüntüleri:** 11,000 (11 farklı parazit türü)
- **Test Görüntüleri:** 2,200
- **Parazit Türleri:** 11 farklı tür

### Eğitimde Kullanılan Sınıflar

1. **Ascaris lumbricoides** (Yuvarlak solucan) - 1,000 örnek
2. **Capillaria philippinensis** - 1,000 örnek
3. **Enterobius vermicularis** (Kıl kurdu) - 1,000 örnek
4. **Fasciolopsis buski** - 553 örnek

**Toplam Eğitim Verisi:** 3,553 görüntü
- **Eğitim Seti:** 2,842 görüntü (%80)
- **Validasyon Seti:** 711 görüntü (%20)

### Veri Ön İşleme

- **Görüntü Boyutu:** 224x224 piksel
- **Normalizasyon:** ImageNet standartlarına göre normalize edilmiştir
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Data Augmentation:** Eğitim sırasında kullanılan teknikler:
  - Random Rotation (±20°)
  - Random Horizontal Flip
  - Random Affine (Translation: ±20%)
  - Color Jitter (Brightness & Contrast: ±20%)

---

## 🏗️ Model Mimarisi

### Kullanılan Yöntem: Transfer Learning (ResNet18)

Transfer Learning, ImageNet'te önceden eğitilmiş ResNet18 modelini kullanarak, az veri ile yüksek performans elde etmeyi sağlar.

### Model Yapısı

```
ResNet18 (Pre-trained on ImageNet)
├── Feature Extractor (Frozen - Dondurulmuş)
│   ├── Conv Layers
│   ├── Batch Normalization
│   └── Residual Blocks
└── Classifier (Fine-tuned - İnce Ayar)
    ├── Global Average Pooling
    ├── Dense(512) + BatchNorm + ReLU + Dropout(0.5)
    ├── Dense(256) + BatchNorm + ReLU + Dropout(0.5)
    └── Dense(4) - Output Layer (Softmax)
```

### Model Özellikleri

- **Toplam Parametre Sayısı:** 11,573,060
- **Eğitilebilir Parametre:** 11,573,060
- **Model Boyutu:** 44.23 MB
- **Input Shape:** (3, 224, 224)
- **Output Shape:** (4,) - 4 sınıf için olasılık dağılımı

### Eğitim Parametreleri

- **Optimizer:** Adam
- **Öğrenme Oranı:** 0.001
- **Batch Size:** 32
- **Epoch Sayısı:** 24 (Early Stopping ile)
- **Loss Function:** Cross Entropy Loss
- **Learning Rate Scheduler:** ReduceLROnPlateau
- **Early Stopping:** Patience = 10

---
)

### Eğitim Sonuçları

- **En İyi Validation Accuracy:** 0.9887 (%98.87)
- **En İyi Validation Loss:** Minimum değere ulaşıldı
- **Early Stopping:** 10 epoch patience sonrası durdu

### Eğitim Grafikleri

`training_history_pytorch.png` dosyasında eğitim sürecinin detaylı grafikleri bulunmaktadır:
- Training ve Validation Accuracy grafikleri
- Training ve Validation Loss grafikleri

---

## 📊 Test Sonuçları ve Performans Metrikleri

### Genel Performans Özeti

| Metrik | Değer | Yüzde | Yorum |
|--------|-------|-------|-------|
| **Accuracy (Doğruluk)** | 0.9887 | **98.87%** | Mükemmel |
| **ROC AUC (Micro-average)** | 0.9994 | 99.94% | Mükemmel |
| **ROC AUC (Macro-average)** | 0.9996 | 99.96% | Mükemmel |

**Değerlendirme:** Model, validasyon seti üzerinde %98.87 doğruluk oranı ile çok yüksek bir performans sergilemektedir. ROC AUC değerlerinin 0.99'un üzerinde olması, modelin sınıfları ayırt etme yeteneğinin mükemmel seviyede olduğunu göstermektedir.

### Sınıf Bazında Detaylı Performans Tablosu

| Parazit Türü | Precision | Recall | F1-Score | ROC AUC | Support |
|--------------|-----------|--------|----------|---------|---------|
| **Ascaris lumbricoides** | 0.9851 | 0.9950 | 0.9900 | 0.9998 | 200 |
| **Capillaria philippinensis** | 0.9851 | 0.9950 | 0.9900 | 0.9996 | 200 |
| **Enterobius vermicularis** | 0.9949 | 0.9700 | 0.9823 | 0.9986 | 200 |
| **Fasciolopsis buski** | 0.9911 | **1.0000** | **0.9955** | **0.9999** | 111 |
| **Macro Average** | 0.9891 | 0.9900 | 0.9895 | 0.9995 | 711 |
| **Weighted Average** | 0.9888 | 0.9887 | 0.9887 | 0.9994 | 711 |

### Metrik Açıklamaları

#### 1. Accuracy (Doğruluk)
- **Tanım:** Genel doğru tahmin oranı
- **Hesaplama:** (Doğru Tahmin Sayısı) / (Toplam Örnek Sayısı)
- **Değerimiz:** 0.9887 (%98.87)
- **Yorum:** 711 örnekten 703'ü doğru tahmin edilmiş, 8'i yanlış

#### 2. Precision (Kesinlik)
- **Tanım:** Modelin pozitif olarak tahmin ettiği örneklerin ne kadarının gerçekten pozitif olduğunu gösterir
- **Formül:** TP / (TP + FP)
- **Yorum:** Model, pozitif tahminlerinde çok kesindir (ortalama 0.99)

#### 3. Recall (Duyarlılık)
- **Tanım:** Gerçek pozitif örneklerin ne kadarının doğru tespit edildiğini gösterir
- **Formül:** TP / (TP + FN)
- **Yorum:** Model, gerçek pozitifleri çok iyi yakalıyor (ortalama 0.99)

#### 4. F1-Score
- **Tanım:** Precision ve Recall'un harmonik ortalamasıdır, dengeli bir performans ölçüsüdür
- **Formül:** 2 × (Precision × Recall) / (Precision + Recall)
- **Yorum:** Tüm sınıflarda yüksek F1-Score değerleri (0.98-1.00)

#### 5. ROC AUC (Area Under the ROC Curve)
- **Tanım:** Modelin sınıfları ayırt etme yeteneğini ölçer
- **Aralık:** 0.0 (kötü) - 1.0 (mükemmel), 0.5 = rastgele
- **Değerlerimiz:** Tüm sınıflar için > 0.99
- **Yorum:** Model, sınıfları ayırt etmede mükemmel performans gösteriyor

#### 6. Support
- **Tanım:** Her sınıftan test edilen örnek sayısı
- **Kullanım:** Ağırlıklı ortalamaların hesaplanmasında kullanılır

---

## 🔍 Confusion Matrix Analizi

### Confusion Matrix Nedir?

Confusion Matrix, modelin her sınıf için yaptığı doğru ve yanlış tahminleri gösteren bir tablodur. Satırlar gerçek sınıfları, sütunlar tahmin edilen sınıfları temsil eder.

### Normalize Edilmemiş Confusion Matrix

`confusion_matrix_detailed.png` dosyasında detaylı görselleştirme bulunmaktadır.

**Confusion Matrix Değerleri:**

```
                    Tahmin Edilen
                 A    C    E    F
Gerçek    A    [199   0    0    1]
          C    [ 0  199    1    0]
          E    [ 3    3  194    0]
          F    [ 0    0    0  111]
```

**Açıklama:**
- **A:** Ascaris lumbricoides
- **C:** Capillaria philippinensis
- **E:** Enterobius vermicularis
- **F:** Fasciolopsis buski

### Hata Analizi

**Toplam Hata:** 8 yanlış tahmin (711 örnekten)

| Sınıf | Doğru | Yanlış | Doğruluk Oranı |
|-------|-------|--------|----------------|
| Ascaris lumbricoides | 199 | 1 | 99.5% |
| Capillaria philippinensis | 199 | 1 | 99.5% |
| Enterobius vermicularis | 194 | 6 | 97.0% |
| Fasciolopsis buski | 111 | 0 | **100%** |

**Önemli Bulgular:**
- **Fasciolopsis buski:** Mükemmel performans - 0 hata
- **Ascaris & Capillaria:** Çok yüksek performans - sadece 1'er hata
- **Enterobius vermicularis:** En fazla hata (6 hata) - bu sınıf diğerleriyle görsel benzerlik gösterebilir

### Normalize Edilmiş Confusion Matrix

`confusion_matrix_normalized.png` dosyasında, confusion matrix yüzde olarak normalize edilmiş halde gösterilmektedir. Bu görselleştirme, her sınıf için doğru tahmin yüzdesini daha net görmemizi sağlar.

**Analiz:**
- Tüm sınıflarda doğru tahmin oranı %97-100 aralığındadır
- En yüksek performans: Fasciolopsis buski (%100)
- En düşük performans: Enterobius vermicularis (%97)

---

## 📊 ROC Eğrileri Analizi

### ROC Eğrisi Nedir?

ROC (Receiver Operating Characteristic) eğrisi, modelin farklı eşik değerlerinde sınıflandırma performansını gösterir. X ekseni False Positive Rate (FPR), Y ekseni True Positive Rate (TPR) olarak gösterilir.

### ROC Eğrileri Görselleştirmesi

`roc_curves.png` dosyasında tüm sınıflar için ROC eğrileri görselleştirilmiştir:
- Her sınıf için ayrı ROC eğrisi
- Micro-average ROC eğrisi
- Macro-average ROC eğrisi
- Rastgele sınıflandırma referans çizgisi (diagonal)

### ROC AUC Değerleri

| Sınıf | ROC AUC | Yorum |
|-------|----------|-------|
| Ascaris lumbricoides | 0.9998 | Mükemmel |
| Capillaria philippinensis | 0.9996 | Mükemmel |
| Enterobius vermicularis | 0.9986 | Mükemmel |
| Fasciolopsis buski | 0.9999 | Mükemmel |
| **Micro-average** | **0.9994** | **Mükemmel** |
| **Macro-average** | **0.9996** | **Mükemmel** |

**Değerlendirme:**
- Tüm sınıflar için ROC AUC değerleri 0.99'un üzerindedir
- Bu, modelin sınıfları ayırt etme yeteneğinin çok yüksek olduğunu gösterir
- Micro ve Macro average değerleri de 0.99'un üzerinde, modelin genel performansının tutarlı olduğunu gösterir
- ROC eğrileri sol üst köşeye yakın, bu mükemmel performansı gösterir

---

## 📉 Precision-Recall Eğrileri

### Precision-Recall Eğrisi Nedir?

Precision-Recall eğrileri, özellikle dengesiz veri setlerinde ROC eğrilerinden daha bilgilendirici olabilir. X ekseni Recall, Y ekseni Precision olarak gösterilir.

### Precision-Recall Eğrileri Görselleştirmesi

`precision_recall_curves.png` dosyasında tüm sınıflar için Precision-Recall eğrileri görselleştirilmiştir.

**Gözlemler:**
- Tüm sınıflar için Precision-Recall eğrileri yüksek değerlerde başlamakta ve yüksek kalarak devam etmektedir
- Bu, modelin hem yüksek precision hem de yüksek recall değerlerine sahip olduğunu gösterir
- Eğriler sağ üst köşeye yakın, bu mükemmel performansı gösterir

---

## 📋 Performans Metrikleri Tablosu

### Görsel Performans Tablosu

`performance_table.png` dosyasında tüm performans metriklerinin görsel tablosu bulunmaktadır. Bu tablo, tüm sınıflar için Precision, Recall, F1-Score, ROC AUC ve Support değerlerini içerir.

### CSV Formatında Metrikler

`performance_metrics.csv` dosyasında tüm metrikler CSV formatında kaydedilmiştir. Bu dosya Excel veya diğer analiz araçlarında açılabilir.

### Sınıf Bazında Karşılaştırma

| Metrik | Ascaris | Capillaria | Enterobius | Fasciolopsis | En İyi |
|--------|---------|------------|------------|--------------|--------|
| **Precision** | 0.9851 | 0.9851 | **0.9949** | 0.9911 | Enterobius |
| **Recall** | 0.9950 | 0.9950 | 0.9700 | **1.0000** | Fasciolopsis |
| **F1-Score** | 0.9900 | 0.9900 | 0.9823 | **0.9955** | Fasciolopsis |
| **ROC AUC** | 0.9998 | 0.9996 | 0.9986 | **0.9999** | Fasciolopsis |

**En İyi Performans Gösteren Sınıf:** Fasciolopsis buski
- En yüksek Recall (1.0000) - Tüm pozitif örnekler doğru tespit edilmiş
- En yüksek F1-Score (0.9955) - En dengeli performans
- En yüksek ROC AUC (0.9999) - En iyi sınıf ayırt etme yeteneği

**En Düşük Performans Gösteren Sınıf:** Enterobius vermicularis
- En düşük Recall (0.9700) - Bazı pozitif örnekler kaçırılmış
- En düşük F1-Score (0.9823) - Diğerlerine göre biraz düşük
- En düşük ROC AUC (0.9986) - Yine de çok yüksek seviyede

**Not:** Tüm sınıflar için performans çok yüksek seviyededir. "En düşük" olarak belirtilen değerler bile 0.97'nin üzerindedir.

---

## 🔬 Detaylı Değerlendirme ve Analiz

### Güçlü Yönler

1. **Yüksek Doğruluk Oranı**
   - %98.87 doğruluk oranı, modelin pratik kullanım için yeterli seviyede olduğunu gösterir
   - Tıbbi uygulamalarda genellikle %95+ doğruluk kabul edilebilir seviye olarak kabul edilir

2. **Tutarlı Performans**
   - Tüm sınıflarda yüksek ve tutarlı performans (Precision, Recall, F1-Score: 0.97-1.00 aralığı)
   - Hiçbir sınıf için düşük performans yok
   - Macro ve Weighted average değerleri birbirine yakın, dengeli performans

3. **Mükemmel ROC AUC Değerleri**
   - Tüm sınıflar için ROC AUC değerleri 0.99'un üzerinde
   - Modelin sınıfları ayırt etme yeteneğinin çok güçlü olduğunu gösterir
   - Micro ve Macro average değerleri de 0.99'un üzerinde

4. **Dengeli Metrikler**
   - Precision ve Recall değerleri birbirine yakın
   - Model hem yanlış pozitif hem de yanlış negatif hatalarından kaçınmaktadır
   - F1-Score değerleri yüksek, dengeli performans

5. **Fasciolopsis buski Mükemmelliği**
   - Bu sınıf için %100 recall değeri
   - Tüm pozitif örneklerin doğru tespit edildiğini gösterir
   - 0 yanlış tahmin - mükemmel performans

### İyileştirme Alanları

1. **Enterobius vermicularis Performansı**
   - Diğer sınıflara göre biraz daha düşük performans (Recall: 0.97)
   - 6 yanlış tahmin (en fazla hata)
   - **Öneriler:**
     - Daha fazla eğitim verisi eklenebilir
     - Data augmentation teknikleri artırılabilir
     - Class weights kullanılabilir (dengesiz veri setleri için)
     - Bu sınıf için özel fine-tuning yapılabilir

2. **Veri Seti Genişletme**
   - Şu anda sadece 4 parazit türü için yeterli veri yüklenebilmiştir
   - Veri setinde toplam 11 parazit türü bulunmaktadır
   - Tüm 11 tür için veri yüklenebilirse, model daha kapsamlı olabilir
   - **Öneriler:**
     - Tüm parazit türleri için veri toplama
     - Veri seti dengesizliğini giderme
     - Daha fazla çeşitlilik sağlama

3. **Ensemble Methods**
   - Birden fazla modelin birleştirilmesi ile performans daha da artırılabilir
   - **Öneriler:**
     - Farklı mimarilerin birleştirilmesi (ResNet, EfficientNet, etc.)
     - Voting veya Stacking yöntemleri
     - Model çeşitliliği sağlama

4. **Hyperparameter Tuning**
   - Öğrenme oranı, batch size gibi hiperparametrelerin optimize edilmesi
   - **Öneriler:**
     - Grid Search veya Random Search
     - Bayesian Optimization
     - Learning rate scheduling optimizasyonu

### Hata Analizi ve Yorumlar

**Toplam Hata:** 8 yanlış tahmin (711 örnekten)

**Hata Dağılımı:**
- **Ascaris lumbricoides:** 1 yanlış tahmin
  - 1 örnek Fasciolopsis buski olarak yanlış tahmin edilmiş
  - Bu iki tür arasında görsel benzerlik olabilir

- **Capillaria philippinensis:** 1 yanlış tahmin
  - 1 örnek Enterobius vermicularis olarak yanlış tahmin edilmiş
  - Bu iki tür arasında görsel benzerlik olabilir

- **Enterobius vermicularis:** 6 yanlış tahmin (en fazla hata)
  - 3 örnek Ascaris lumbricoides olarak yanlış tahmin edilmiş
  - 3 örnek Capillaria philippinensis olarak yanlış tahmin edilmiş
  - Bu, Enterobius vermicularis'in diğer türlerle görsel benzerliklerinin daha fazla olduğunu düşündürmektedir
  - Bu sınıf için daha fazla eğitim verisi veya özel teknikler gerekebilir

- **Fasciolopsis buski:** 0 yanlış tahmin (mükemmel)
  - Tüm örnekler doğru tahmin edilmiş
  - Bu türün diğerlerinden daha ayırt edici özelliklere sahip olduğu düşünülebilir

**Yorum:** Enterobius vermicularis sınıfında daha fazla hata görülmektedir. Bu, bu parazit türünün diğer türlerle görsel benzerliklerinin daha fazla olabileceğini düşündürmektedir. Bu sınıf için özel iyileştirmeler yapılabilir.

---

## 🎯 Model Karşılaştırması ve Yöntem Değerlendirmesi

### Kullanılan Yöntem: Transfer Learning (ResNet18)

#### Avantajlar

1. **Yüksek Performans**
   - %98.87 doğruluk oranı ile çok başarılı sonuçlar
   - Tüm sınıflarda tutarlı yüksek performans
   - Mükemmel ROC AUC değerleri (>0.99)

2. **Hızlı Eğitim**
   - Pre-trained model kullanımı sayesinde daha az epoch ile yüksek performans
   - 24 epoch'ta optimal performansa ulaşıldı
   - Toplam eğitim süresi: ~26 dakika (CPU'da)

3. **Genelleştirme**
   - ImageNet'te öğrenilmiş özellikler, tıbbi görüntülerde de etkili
   - Transfer Learning sayesinde az veri ile yüksek performans
   - Overfitting riski düşük

4. **Stabilite**
   - Tüm sınıflarda tutarlı yüksek performans
   - Early stopping ile optimal noktada durdu
   - Learning rate scheduling ile stabil eğitim

#### Dezavantajlar

1. **Model Boyutu**
   - 44.23 MB (görece büyük)
   - Mobil uygulamalar için optimize edilebilir
   - Model quantization veya pruning uygulanabilir

2. **Hesaplama Gereksinimi**
   - CPU'da eğitim uzun sürebilir (GPU önerilir)
   - Inference sırasında da GPU kullanımı hızlandırır
   - Edge device'lar için optimize edilmiş modeller kullanılabilir

### Alternatif Yöntem: Temel CNN

Temel CNN modeli de test edilebilir, ancak genellikle:
- Daha uzun eğitim süresi gerektirir
- Daha fazla veriye ihtiyaç duyar
- Transfer Learning'e göre daha düşük performans gösterebilir
- Sıfırdan öğrenme gerektirir

**Sonuç:** Transfer Learning (ResNet18) yöntemi, bu proje için en uygun seçimdir. Az veri ile yüksek performans sağlamıştır.

---

## 📁 Oluşturulan Dosyalar ve Görselleştirmeler

### Model Dosyaları

- `models/parasite_model_pytorch.pth` - Eğitilmiş model (44.23 MB)
- `models/class_names_pytorch.pkl` - Sınıf isimleri
- `models/training_history_pytorch.json` - Eğitim geçmişi (JSON formatında)

### Görselleştirmeler (Ana Klasörde)

1. **`confusion_matrix_detailed.png`** - Detaylı confusion matrix
   - Her sınıf için doğru ve yanlış tahminleri gösterir
   - Sayısal değerlerle birlikte

2. **`confusion_matrix_normalized.png`** - Normalize edilmiş confusion matrix
   - Yüzde olarak normalize edilmiş
   - Her sınıf için doğru tahmin yüzdesini daha net gösterir

3. **`roc_curves.png`** - ROC eğrileri
   - Tüm sınıflar için ayrı ROC eğrileri
   - Micro ve Macro average eğrileri
   - Rastgele sınıflandırma referans çizgisi

4. **`precision_recall_curves.png`** - Precision-Recall eğrileri
   - Her sınıf için Precision-Recall performansı
   - Dengesiz veri setleri için önemli metrik

5. **`performance_table.png`** - Performans metrikleri tablosu
   - Tüm metriklerin görsel tablosu
   - Renkli ve okunabilir format

6. **`training_history_pytorch.png`** - Eğitim geçmişi grafikleri
   - Training ve Validation Accuracy grafikleri
   - Training ve Validation Loss grafikleri

### Sonuç Dosyaları

- `performance_metrics.csv` - CSV formatında performans metrikleri
- `detailed_results.json` - JSON formatında detaylı sonuçlar
- `test_predictions_pytorch.json` - Test seti tahminleri ve olasılık değerleri
- `DETAYLI_SONUCLAR.md` - Kapsamlı sonuç analizi ve değerlendirme raporu
- `SONUCLAR.md` - Özet test sonuçları raporu

### Eğitim Logları

- `training_log.txt` - Eğitim süreci log dosyası
- `EGITIM_TAMAMLANDI.txt` - Eğitim tamamlanma bildirimi

---

## 🚀 Kurulum ve Kullanım

### 1. Python Kurulumu

**ÖNEMLİ:** Python 3.14 için TensorFlow desteklenmiyor. Bu proje **PyTorch** kullanmaktadır.

Python 3.8-3.14 arası sürümler çalışır. [Python'u buradan indirebilirsiniz](https://www.python.org/downloads/).

### 2. Gerekli Kütüphaneleri Yükleyin

**PyTorch versiyonu için:**
```bash
python -m pip install torch torchvision torchaudio scikit-learn matplotlib seaborn pandas pillow numpy
```

Veya Windows'ta Python 3.14 için:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install scikit-learn matplotlib seaborn pandas pillow numpy
```

### 3. Veri Setini Hazırlayın

`Chula-ParasiteEgg-11.zip` dosyasının proje klasöründe olduğundan emin olun. Model eğitimi sırasında otomatik olarak çıkarılacaktır.

### 4. Model Eğitimi

#### Transfer Learning ile Eğitim (Önerilen):
```bash
python train_pytorch.py --transfer
```

#### Temel CNN Modeli ile Eğitim:
```bash
python train_pytorch.py
```

#### Özelleştirilmiş Parametreler:
```bash
python train_pytorch.py --transfer --epochs 100 --batch_size 64 --lr 0.0001
```

**Parametreler:**
- `--zip_path`: ZIP dosyası yolu (varsayılan: `Chula-ParasiteEgg-11.zip`)
- `--transfer`: Transfer learning kullan (ResNet18)
- `--epochs`: Epoch sayısı (varsayılan: 50)
- `--batch_size`: Batch boyutu (varsayılan: 32)
- `--lr`: Öğrenme oranı (varsayılan: 0.001)

#### Kolay Başlatma (Windows):
`run.bat` dosyasını çift tıklayarak tüm süreci başlatabilirsiniz.

### 5. Model Değerlendirme

Eğitilmiş modeli test verileriyle değerlendirmek için:
```bash
python evaluate_pytorch.py
```

### 6. Detaylı Sonuç Analizi

Detaylı görselleştirmeler ve analizler için:
```bash
python create_detailed_results.py
```

Bu komut şunları oluşturur:
- Detaylı confusion matrix
- ROC eğrileri
- Precision-Recall eğrileri
- Performans metrikleri tablosu
- CSV ve JSON formatında sonuçlar

### 7. Tam Pipeline (Eğitim + Değerlendirme):
```bash
python run_pytorch.py
```

---

## 📁 Proje Yapısı

```
parazit/
├── Chula-ParasiteEgg-11.zip      # Veri seti
├── data/                          # Çıkarılan veriler (otomatik oluşturulur)
│   ├── Chula-ParasiteEgg-11/     # Eğitim verileri
│   └── _test/                     # Test verileri
├── models/                        # Eğitilmiş modeller (otomatik oluşturulur)
│   ├── parasite_model_pytorch.pth # Eğitilmiş model
│   ├── class_names_pytorch.pkl   # Sınıf isimleri
│   └── training_history_pytorch.json # Eğitim geçmişi
├── data_loader_pytorch.py        # Veri yükleme modülü (PyTorch)
├── model_pytorch.py              # Model mimarisi (PyTorch)
├── train_pytorch.py              # Eğitim scripti (PyTorch)
├── evaluate_pytorch.py           # Değerlendirme scripti (PyTorch)
├── create_detailed_results.py    # Detaylı sonuç analizi scripti
├── run_pytorch.py                # Tam pipeline scripti
├── check_training_status.py      # Eğitim durumu kontrol scripti
├── check_completion.py           # Tamamlanma kontrol scripti
├── monitor_training.ps1          # Eğitim izleme scripti (PowerShell)
├── start_training.bat            # Windows batch dosyası
├── run.bat                        # Kolay başlatma dosyası
├── requirements.txt              # Gerekli kütüphaneler
├── README.md                     # Bu dosya
├── DETAYLI_SONUCLAR.md           # Kapsamlı sonuç analizi
├── SONUCLAR.md                   # Özet test sonuçları
├── confusion_matrix_detailed.png # Detaylı confusion matrix
├── confusion_matrix_normalized.png # Normalize edilmiş confusion matrix
├── roc_curves.png                # ROC eğrileri
├── precision_recall_curves.png   # Precision-Recall eğrileri
├── performance_table.png         # Performans metrikleri tablosu
├── training_history_pytorch.png  # Eğitim geçmişi grafikleri
├── performance_metrics.csv       # CSV formatında metrikler
├── detailed_results.json         # JSON formatında detaylı sonuçlar
└── test_predictions_pytorch.json # Test seti tahminleri
```

---

## 🔧 Teknik Detaylar

### Kullanılan Teknolojiler

- **Python:** 3.14.1
- **PyTorch:** 2.9.1+cpu
- **Torchvision:** 0.24.1+cpu
- **NumPy:** 2.3.3
- **Pillow:** 11.3.0
- **scikit-learn:** 1.7.2
- **matplotlib:** 3.10.7
- **seaborn:** 0.13.2
- **pandas:** 2.3.3

### Hesaplama Ortamı

- **Cihaz:** CPU
- **İşletim Sistemi:** Windows 10
- **Eğitim Süresi:** ~26 dakika (24 epoch)
- **Inference Süresi:** ~2-3 saniye (711 örnek için)

### Metrik Hesaplama Yöntemleri

Tüm metrikler scikit-learn kütüphanesi kullanılarak hesaplanmıştır:
- `accuracy_score()` - Accuracy hesaplama
- `precision_score()` - Precision hesaplama
- `recall_score()` - Recall hesaplama
- `f1_score()` - F1-Score hesaplama
- `roc_curve()` - ROC eğrisi hesaplama
- `auc()` - ROC AUC hesaplama
- `confusion_matrix()` - Confusion matrix hesaplama
- `classification_report()` - Detaylı sınıflandırma raporu

---

## 📈 Sonuç ve Öneriler

### Genel Değerlendirme

Model, mikroskop görüntülerinden parazit yumurtası tespiti için **çok başarılı** sonuçlar vermiştir. %98.87 doğruluk oranı ve tüm sınıflarda yüksek performans metrikleri, modelin gerçek dünya uygulamalarında kullanılabilecek seviyede olduğunu göstermektedir.

### Pratik Kullanım Önerileri

1. **Tıbbi Laboratuvarlar**
   - Model, rutin parazit analizlerinde yardımcı bir araç olarak kullanılabilir
   - İnsan uzmanları ile birlikte çalışarak analiz süresini kısaltabilir
   - Özellikle yüksek hacimli laboratuvarlarda faydalı olabilir

2. **Eğitim**
   - Tıp öğrencilerine parazit türlerini öğretmek için interaktif bir öğrenme aracı olarak kullanılabilir
   - Görsel örneklerle öğrenmeyi destekler
   - Anında geri bildirim sağlar

3. **Araştırma**
   - Epidemiyolojik çalışmalarda otomatik parazit sayımı ve sınıflandırması için kullanılabilir
   - Büyük veri setlerinin hızlı analizi
   - Tutarlı ve objektif sonuçlar

### Gelecek Çalışmalar

1. **Veri Seti Genişletme**
   - Tüm 11 parazit türü için yeterli veri toplanması
   - Veri seti dengesizliğinin giderilmesi
   - Daha fazla çeşitlilik sağlanması

2. **Model Optimizasyonu**
   - Hyperparameter tuning ile performansın daha da artırılması
   - Model quantization ile boyutun küçültülmesi
   - Inference hızının optimize edilmesi

3. **Ensemble Learning**
   - Birden fazla modelin birleştirilmesi
   - Farklı mimarilerin kombinasyonu
   - Voting veya Stacking yöntemleri

4. **Real-time Uygulama**
   - Mikroskop görüntülerinin gerçek zamanlı analizi için optimizasyon
   - Edge device'lar için model optimizasyonu
   - Web veya mobil uygulama geliştirme

---

## 🔧 Sorun Giderme

### GPU Kullanımı
Model otomatik olarak GPU'yu algılar ve kullanır. GPU yoksa CPU kullanılır. GPU kullanımı eğitim süresini önemli ölçüde kısaltır.

### Bellek Hatası
Batch size'ı küçültün:
```bash
python train_pytorch.py --batch_size 16
```

### Veri Yükleme Hatası
ZIP dosyasının doğru konumda olduğundan emin olun. Veri klasörü otomatik olarak oluşturulacaktır.

### Unicode Hatası
Windows'ta bazı karakterler sorun çıkarabilir. Scriptler Unicode karakterlerden kaçınacak şekilde güncellenmiştir.

---

## 📝 Notlar

- İlk eğitimde ZIP dosyası otomatik olarak çıkarılır (biraz zaman alabilir)
- Model eğitimi GPU ile daha hızlıdır (CPU'da ~26 dakika, GPU'da ~5-10 dakika)
- Early stopping kullanılır, bu yüzden model erken durabilir (optimal performansa ulaşıldığında)
- Data augmentation kullanılır (rotation, shift, flip, zoom)
- Tüm görselleştirmeler ana klasörde oluşturulur
- Detaylı sonuçlar için `DETAYLI_SONUCLAR.md` dosyasına bakınız

---

## 👨‍💻 Geliştirici

Tıbbi Görüntü İşleme Dersi - Ödev Projesi  
**Tarih:** 5 Aralık 2025

---

## 📄 Lisans

Bu proje eğitim amaçlıdır. Chula-ParasiteEgg-11 veri seti Attribution 4.0 International License altında lisanslanmıştır.

---

## 📚 Referanslar ve Kaynaklar

- **Veri Seti:** Chula-ParasiteEgg-11 - https://icip2022challenge.piclab.ai/
- **PyTorch:** https://pytorch.org/
- **ResNet18:** Deep Residual Learning for Image Recognition (He et al., 2015)
- **scikit-learn:** https://scikit-learn.org/

---

**Son Güncelleme:** 5 Aralık 2025  
**Model Versiyonu:** Transfer Learning (ResNet18)  
**Test Seti:** 711 validasyon örneği  
**Doğruluk Oranı:** %98.87
