"""
Parazit Yumurta Tespiti - Model Eğitimi (PyTorch Versiyonu)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from data_loader_pytorch import ParasiteDataLoader
from model_pytorch import create_cnn_model, create_transfer_learning_model, get_model_summary
import json
from datetime import datetime


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validasyon"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    zip_path='Chula-ParasiteEgg-11.zip',
    use_transfer_learning=False,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    img_size=(224, 224),
    model_save_path='models/parasite_model_pytorch.pth'
):
    """Modeli eğitir"""
    print("=" * 60)
    print("PARAZİT YUMURTA TESPİTİ - MODEL EĞİTİMİ (PyTorch)")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nKullanılan cihaz: {device}")
    
    # Veri yükleyici
    print("\n" + "=" * 60)
    print("VERİ YÜKLEME")
    print("=" * 60)
    data_loader = ParasiteDataLoader(zip_path, img_size=img_size, batch_size=batch_size)
    data_loader.extract_zip(extract_to='data')
    
    X_train, X_val, y_train, y_val = data_loader.prepare_training_data(data_dir='data')
    train_loader, val_loader = data_loader.get_data_loaders(X_train, X_val, y_train, y_val)
    
    # Model
    print("\n" + "=" * 60)
    print("MODEL OLUŞTURMA")
    print("=" * 60)
    num_classes = len(data_loader.class_names)
    
    if use_transfer_learning:
        print("Transfer Learning modeli oluşturuluyor (ResNet18)...")
        model = create_transfer_learning_model(num_classes)
    else:
        print("CNN modeli oluşturuluyor...")
        model = create_cnn_model(num_classes)
    
    model = model.to(device)
    get_model_summary(model)
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Eğitim
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ")
    print("=" * 60)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': data_loader.class_names,
                'num_classes': num_classes,
            }, model_save_path)
            print(f'[OK] En iyi model kaydedildi! (Val Acc: {val_acc:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping! (Patience: {patience})')
            break
    
    # Geçmişi kaydet
    import pickle
    with open('models/class_names_pytorch.pkl', 'wb') as f:
        pickle.dump(data_loader.class_names, f)
    
    with open('models/training_history_pytorch.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 60)
    print(f"En iyi validation accuracy: {best_val_acc:.4f}")
    
    # Tamamlandı bildirimi dosyası oluştur
    completion_file = "EGITIM_TAMAMLANDI.txt"
    with open(completion_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EĞİTİM TAMAMLANDI!\n")
        f.write("=" * 60 + "\n")
        f.write(f"En iyi validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)\n")
        f.write(f"Toplam epoch sayısı: {len(history['train_acc'])}\n")
        f.write(f"Model dosyası: {model_save_path}\n")
        f.write(f"Tamamlanma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nBildirim dosyasi olusturuldu: {completion_file}")
    print("Terminal'de 'EGITIM TAMAMLANDI' yazisini goreceksiniz!")
    
    return model, history, data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parazit Yumurta Tespiti Model Eğitimi (PyTorch)')
    parser.add_argument('--zip_path', type=str, default='Chula-ParasiteEgg-11.zip')
    parser.add_argument('--transfer', action='store_true', help='Transfer learning kullan')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    train_model(
        zip_path=args.zip_path,
        use_transfer_learning=args.transfer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

