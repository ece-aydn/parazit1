"""
Detaylı sonuç analizi ve görselleştirme scripti
ROC eğrileri, confusion matrix, performans tabloları oluşturur
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
import pickle
from data_loader_pytorch import ParasiteDataLoader, ParasiteDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model_pytorch import create_transfer_learning_model
import os

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_model_and_data():
    """Modeli ve verileri yükler"""
    print("Model yükleniyor...")
    model_path = 'models/parasite_model_pytorch.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    
    model = create_transfer_learning_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Validasyon verilerini yükle
    data_loader = ParasiteDataLoader('Chula-ParasiteEgg-11.zip', img_size=(224, 224))
    data_loader.class_names = class_names
    data_loader.label_encoder.fit(class_names)
    
    X_train, X_val, y_train, y_val = data_loader.prepare_training_data(data_dir='data')
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = ParasiteDataset(X_val, y_val, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return model, val_loader, class_names, y_val, device

def get_predictions_and_probs(model, val_loader, device):
    """Tahminler ve olasılıkları alır"""
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def plot_confusion_matrix_detailed(y_true, y_pred, class_names, save_path='confusion_matrix_detailed.png'):
    """Detaylı confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize edilmiş confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Örnek Sayısı'})
    
    ax.set_title('Confusion Matrix - Karışıklık Matrisi', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Gerçek Sınıf (True Label)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tahmin Edilen Sınıf (Predicted Label)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix kaydedildi: {save_path}")
    plt.close()
    
    # Normalize edilmiş confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Yüzde (%)'})
    
    ax.set_title('Normalized Confusion Matrix - Normalize Edilmiş Karışıklık Matrisi', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Gerçek Sınıf (True Label)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tahmin Edilen Sınıf (Predicted Label)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix kaydedildi: confusion_matrix_normalized.png")
    plt.close()
    
    return cm, cm_normalized

def plot_roc_curves(y_true, y_probs, class_names, save_path='roc_curves.png'):
    """ROC eğrileri"""
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Her sınıf için ROC eğrisi
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='black', linestyle='--', lw=2,
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    
    # Macro-average ROC
    fpr_macro = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_probs[:, i])[0] 
                                         for i in range(len(class_names))]))
    tpr_macro = np.zeros_like(fpr_macro)
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        tpr_macro += np.interp(fpr_macro, fpr, tpr)
    tpr_macro /= len(class_names)
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    ax.plot(fpr_macro, tpr_macro, color='navy', linestyle='--', lw=2,
            label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Yanlış Pozitif Oranı)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Doğru Pozitif Oranı)', fontsize=12, fontweight='bold')
    ax.set_title('ROC Eğrileri - Receiver Operating Characteristic Curves', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC eğrileri kaydedildi: {save_path}")
    plt.close()
    
    return roc_auc_micro, roc_auc_macro

def plot_precision_recall_curves(y_true, y_probs, class_names, save_path='precision_recall_curves.png'):
    """Precision-Recall eğrileri"""
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        
        ax.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{class_name} (AP = {ap:.3f})')
    
    ax.set_xlabel('Recall (Duyarlılık)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (Kesinlik)', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Eğrileri', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall eğrileri kaydedildi: {save_path}")
    plt.close()

def create_performance_table(y_true, y_pred, y_probs, class_names):
    """Performans tablosu oluşturur"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Genel metrikler
    accuracy = accuracy_score(y_true, y_pred)
    
    # Sınıf bazında metrikler
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Support (her sınıftan örnek sayısı)
    cm = confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1)
    
    # ROC AUC hesapla
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    roc_aucs = []
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_aucs.append(auc(fpr, tpr))
    
    # Tablo oluştur
    data = {
        'Parazit Türü': class_names,
        'Precision': [f'{p:.4f}' for p in precision],
        'Recall': [f'{r:.4f}' for r in recall],
        'F1-Score': [f'{f:.4f}' for f in f1],
        'ROC AUC': [f'{a:.4f}' for a in roc_aucs],
        'Support': support.tolist()
    }
    
    df = pd.DataFrame(data)
    
    # Genel ortalamalar
    macro_avg = {
        'Parazit Türü': 'Macro Average',
        'Precision': f'{precision.mean():.4f}',
        'Recall': f'{recall.mean():.4f}',
        'F1-Score': f'{f1.mean():.4f}',
        'ROC AUC': f'{np.mean(roc_aucs):.4f}',
        'Support': support.sum()
    }
    
    weighted_avg = {
        'Parazit Türü': 'Weighted Average',
        'Precision': f'{np.average(precision, weights=support):.4f}',
        'Recall': f'{np.average(recall, weights=support):.4f}',
        'F1-Score': f'{np.average(f1, weights=support):.4f}',
        'ROC AUC': f'{np.average(roc_aucs, weights=support):.4f}',
        'Support': support.sum()
    }
    
    df = pd.concat([df, pd.DataFrame([macro_avg, weighted_avg])], ignore_index=True)
    
    return df, accuracy

def plot_performance_table(df, save_path='performance_table.png'):
    """Performans tablosunu görselleştirir"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Başlık stil
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Son 2 satırı (ortalama) vurgula
    for i in range(len(df.columns)):
        table[(len(df)-2, i)].set_facecolor('#FFE082')
        table[(len(df)-1, i)].set_facecolor('#FFE082')
    
    plt.title('Sınıflandırma Performans Metrikleri Tablosu', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performans tablosu kaydedildi: {save_path}")
    plt.close()

def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print("DETAYLI SONUÇ ANALİZİ VE GÖRSELLEŞTİRME")
    print("=" * 70)
    
    # Model ve verileri yükle
    model, val_loader, class_names, y_val, device = load_model_and_data()
    
    # Tahminler ve olasılıklar
    print("\nTahminler yapılıyor...")
    y_pred, y_probs, y_true = get_predictions_and_probs(model, val_loader, device)
    
    # Confusion Matrix
    print("\nConfusion Matrix oluşturuluyor...")
    cm, cm_normalized = plot_confusion_matrix_detailed(y_true, y_pred, class_names)
    
    # ROC Eğrileri
    print("\nROC eğrileri oluşturuluyor...")
    roc_auc_micro, roc_auc_macro = plot_roc_curves(y_true, y_probs, class_names)
    
    # Precision-Recall Eğrileri
    print("\nPrecision-Recall eğrileri oluşturuluyor...")
    plot_precision_recall_curves(y_true, y_probs, class_names)
    
    # Performans Tablosu
    print("\nPerformans tablosu oluşturuluyor...")
    df, accuracy = create_performance_table(y_true, y_pred, y_probs, class_names)
    
    # Tabloyu CSV olarak kaydet
    df.to_csv('performance_metrics.csv', index=False, encoding='utf-8-sig')
    print("Performans metrikleri CSV olarak kaydedildi: performance_metrics.csv")
    
    # Tabloyu görselleştir
    plot_performance_table(df)
    
    # Sonuçları JSON olarak kaydet
    results = {
        'accuracy': float(accuracy),
        'roc_auc_micro': float(roc_auc_micro),
        'roc_auc_macro': float(roc_auc_macro),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names.tolist() if isinstance(class_names, np.ndarray) else class_names,
        'performance_metrics': df.to_dict('records')
    }
    
    with open('detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Detaylı sonuçlar JSON olarak kaydedildi: detailed_results.json")
    
    # Özet yazdır
    print("\n" + "=" * 70)
    print("ÖZET SONUÇLAR")
    print("=" * 70)
    print(f"Accuracy (Doğruluk): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC AUC (Micro-average): {roc_auc_micro:.4f}")
    print(f"ROC AUC (Macro-average): {roc_auc_macro:.4f}")
    print("\nPerformans Tablosu:")
    print(df.to_string(index=False))
    print("\n" + "=" * 70)
    print("Tüm görselleştirmeler ve tablolar oluşturuldu!")
    print("=" * 70)

if __name__ == '__main__':
    main()





