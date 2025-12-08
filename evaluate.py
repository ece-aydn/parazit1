"""
Parazit Yumurta Tespiti - Model Değerlendirme
Test verileriyle modeli test eder ve doğruluk oranını ölçer
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from data_loader import ParasiteDataLoader


def load_model_and_classes(model_path='models/parasite_model.h5', 
                          class_names_path='models/class_names.pkl'):
    """
    Eğitilmiş modeli ve sınıf isimlerini yükler
    
    Returns:
        model: Yüklenen model
        class_names: Sınıf isimleri
    """
    print(f"Model yükleniyor: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"Sınıf isimleri yükleniyor: {class_names_path}")
    with open(class_names_path, 'rb') as f:
        class_names = pickle.load(f)
    
    print(f"Yüklenen sınıf sayısı: {len(class_names)}")
    print(f"Sınıflar: {class_names}")
    
    return model, class_names


def evaluate_on_validation(model, data_loader, data_dir='data'):
    """
    Validasyon seti üzerinde değerlendirme yapar
    
    Returns:
        accuracy: Doğruluk oranı
        report: Classification report
        cm: Confusion matrix
    """
    print("\n" + "=" * 60)
    print("VALIDASYON SETİ DEĞERLENDİRMESİ")
    print("=" * 60)
    
    # Validasyon verilerini yükle
    X_train, X_val, y_train, y_val = data_loader.prepare_training_data(data_dir=data_dir)
    
    # Tahmin yap
    print("Tahminler yapılıyor...")
    y_pred_proba = model.predict(X_val, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Doğruluk hesapla
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nValidasyon Doğruluk Oranı: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    report = classification_report(
        y_val, y_pred,
        target_names=data_loader.class_names,
        output_dict=True
    )
    
    print("\nSınıf Bazında Performans:")
    print(classification_report(
        y_val, y_pred,
        target_names=data_loader.class_names
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    return accuracy, report, cm, y_val, y_pred


def evaluate_on_test(model, data_loader, class_names, data_dir='data'):
    """
    Test seti üzerinde değerlendirme yapar
    
    Not: Test setinde gerçek etiketler yok, bu yüzden sadece tahminler yapılır
    """
    print("\n" + "=" * 60)
    print("TEST SETİ DEĞERLENDİRMESİ")
    print("=" * 60)
    
    # Test verilerini yükle
    X_test = data_loader.prepare_test_data(data_dir=data_dir)
    
    # Tahmin yap
    print("Test görüntüleri üzerinde tahminler yapılıyor...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Tahmin edilen sınıfları göster
    predicted_classes = [class_names[i] for i in y_pred]
    
    # Sınıf dağılımını göster
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nTahmin Edilen Sınıf Dağılımı:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {class_names[cls_idx]}: {count} görüntü ({count/len(y_pred)*100:.2f}%)")
    
    # Tahminleri kaydet
    results = {
        'predictions': y_pred.tolist(),
        'predicted_classes': predicted_classes,
        'probabilities': y_pred_proba.tolist()
    }
    
    import json
    with open('test_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nTest tahminleri kaydedildi: test_predictions.json")
    
    return y_pred, y_pred_proba


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Confusion matrix'i görselleştirir
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix kaydedildi: {save_path}")
    plt.close()


def plot_training_history(history_path='models/training_history.json',
                         save_path='training_history.png'):
    """
    Eğitim geçmişini görselleştirir
    """
    import json
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', marker='o')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['loss'], label='Training Loss', marker='o')
    axes[1].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Eğitim geçmişi grafiği kaydedildi: {save_path}")
    plt.close()


def main():
    """
    Ana değerlendirme fonksiyonu
    """
    print("=" * 60)
    print("PARAZİT YUMURTA TESPİTİ - MODEL DEĞERLENDİRME")
    print("=" * 60)
    
    # Model ve sınıf isimlerini yükle
    model, class_names = load_model_and_classes()
    
    # Veri yükleyiciyi oluştur
    data_loader = ParasiteDataLoader('Chula-ParasiteEgg-11.zip', img_size=(224, 224))
    data_loader.class_names = class_names
    data_loader.label_encoder.fit(class_names)
    
    # Validasyon seti üzerinde değerlendirme
    val_accuracy, report, cm, y_val, y_pred = evaluate_on_validation(model, data_loader)
    
    # Confusion matrix görselleştir
    plot_confusion_matrix(cm, class_names)
    
    # Eğitim geçmişini görselleştir
    if os.path.exists('models/training_history.json'):
        plot_training_history()
    
    # Test seti üzerinde tahmin yap
    test_predictions, test_proba = evaluate_on_test(model, data_loader, class_names)
    
    # Özet rapor
    print("\n" + "=" * 60)
    print("DEĞERLENDİRME ÖZETİ")
    print("=" * 60)
    print(f"Validasyon Doğruluk Oranı: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Test görüntü sayısı: {len(test_predictions)}")
    print("\nDeğerlendirme tamamlandı!")
    print("=" * 60)


if __name__ == '__main__':
    main()






