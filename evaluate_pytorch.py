"""
Parazit Yumurta Tespiti - Model Değerlendirme (PyTorch Versiyonu)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from data_loader_pytorch import ParasiteDataLoader, ParasiteDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model_pytorch import create_cnn_model, create_transfer_learning_model


def load_model_and_classes(model_path='models/parasite_model_pytorch.pth'):
    """Eğitilmiş modeli yükler"""
    print(f"Model yükleniyor: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    
    # Model oluştur (transfer learning kullanılıp kullanılmadığını tahmin et)
    try:
        model = create_transfer_learning_model(num_classes)
    except:
        model = create_cnn_model(num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Yüklenen sınıf sayısı: {len(class_names)}")
    print(f"Sınıflar: {class_names}")
    
    return model, class_names


def evaluate_on_validation(model, data_loader, device, data_dir='data'):
    """Validasyon seti üzerinde değerlendirme"""
    print("\n" + "=" * 60)
    print("VALIDASYON SETİ DEĞERLENDİRMESİ")
    print("=" * 60)
    
    X_train, X_val, y_train, y_val = data_loader.prepare_training_data(data_dir=data_dir)
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = ParasiteDataset(X_val, y_val, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nValidasyon Doğruluk Oranı: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    report = classification_report(all_labels, all_preds, target_names=data_loader.class_names, output_dict=True)
    print("\nSınıf Bazında Performans:")
    print(classification_report(all_labels, all_preds, target_names=data_loader.class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm, all_labels, all_preds


def evaluate_on_test(model, data_loader, class_names, device, data_dir='data'):
    """Test seti üzerinde değerlendirme"""
    print("\n" + "=" * 60)
    print("TEST SETİ DEĞERLENDİRMESİ")
    print("=" * 60)
    
    X_test = data_loader.prepare_test_data(data_dir=data_dir)
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ParasiteDataset(X_test, labels=None, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    predicted_classes = [class_names[i] for i in all_preds]
    
    unique, counts = np.unique(all_preds, return_counts=True)
    print("\nTahmin Edilen Sınıf Dağılımı:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {class_names[cls_idx]}: {count} görüntü ({count/len(all_preds)*100:.2f}%)")
    
    results = {
        'predictions': [int(p) for p in all_preds],  # NumPy int64'ü Python int'e çevir
        'predicted_classes': predicted_classes,
        'probabilities': [[float(prob) for prob in probs] for probs in all_probs]  # NumPy array'leri list'e çevir
    }
    
    with open('test_predictions_pytorch.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nTest tahminleri kaydedildi: test_predictions_pytorch.json")
    
    return all_preds, all_probs


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_pytorch.png'):
    """Confusion matrix görselleştirir"""
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


def plot_training_history(history_path='models/training_history_pytorch.json',
                         save_path='training_history_pytorch.png'):
    """Eğitim geçmişini görselleştirir"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_acc'], label='Training Accuracy', marker='o')
    axes[0].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_loss'], label='Training Loss', marker='o')
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
    """Ana değerlendirme fonksiyonu"""
    print("=" * 60)
    print("PARAZİT YUMURTA TESPİTİ - MODEL DEĞERLENDİRME (PyTorch)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    model, class_names = load_model_and_classes()
    model = model.to(device)
    
    data_loader = ParasiteDataLoader('Chula-ParasiteEgg-11.zip', img_size=(224, 224))
    data_loader.class_names = class_names
    data_loader.label_encoder.fit(class_names)
    
    val_accuracy, report, cm, y_val, y_pred = evaluate_on_validation(model, data_loader, device)
    plot_confusion_matrix(cm, class_names)
    
    if os.path.exists('models/training_history_pytorch.json'):
        plot_training_history()
    
    test_predictions, test_proba = evaluate_on_test(model, data_loader, class_names, device)
    
    print("\n" + "=" * 60)
    print("DEĞERLENDİRME ÖZETİ")
    print("=" * 60)
    print(f"Validasyon Doğruluk Oranı: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Test görüntü sayısı: {len(test_predictions)}")
    print("\nDeğerlendirme tamamlandı!")
    print("=" * 60)


if __name__ == '__main__':
    main()

