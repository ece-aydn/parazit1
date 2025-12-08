"""
Parazit Yumurta Tespiti - Veri Yükleme Modülü (PyTorch Versiyonu)
"""

import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ParasiteDataset(Dataset):
    """PyTorch Dataset sınıfı"""
    
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # NumPy array olarak tut
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Normalize edilmişse [0, 255]'e çevir
        if isinstance(image, np.ndarray) and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, torch.tensor(self.labels[idx], dtype=torch.long)
        return image


class ParasiteDataLoader:
    """Parazit görüntü verilerini yükleyen ve işleyen sınıf"""
    
    def __init__(self, zip_path, img_size=(224, 224), batch_size=32):
        self.zip_path = zip_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def extract_zip(self, extract_to='data'):
        """ZIP dosyasını çıkarır"""
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
            print(f"ZIP dosyası çıkarılıyor: {self.zip_path}")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print("ZIP dosyası başarıyla çıkarıldı!")
        else:
            print(f"Veri klasörü zaten mevcut: {extract_to}")
    
    def load_images_from_folder(self, folder_path, label_prefix=''):
        """Klasörden görüntüleri ve etiketlerini yükler"""
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"Klasör bulunamadı: {folder_path}")
            return np.array(images), np.array(labels)
        
        data_path = os.path.join(folder_path, 'data')
        if not os.path.exists(data_path):
            data_path = folder_path
        
        print(f"Görüntüler yükleniyor: {data_path}")
        
        for filename in os.listdir(data_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(data_path, filename)
                
                try:
                    # Kesik görüntüleri kontrol et
                    img = Image.open(img_path)
                    img.verify()  # Görüntüyü doğrula
                    img = Image.open(img_path)  # Tekrar aç (verify dosyayı kapatır)
                    img = img.convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    
                    images.append(img_array)
                    
                    if label_prefix:
                        label = filename.rsplit('_', 1)[0]
                        labels.append(label)
                    else:
                        labels.append('unknown')
                        
                except Exception as e:
                    # Hatalı görüntüleri atla
                    continue
        
        print(f"Toplam {len(images)} görüntü yüklendi")
        return np.array(images), np.array(labels)
    
    def prepare_training_data(self, data_dir='data'):
        """Eğitim verilerini hazırlar"""
        train_folder = os.path.join(data_dir, 'Chula-ParasiteEgg-11', 'Chula-ParasiteEgg-11', 'data')
        
        if not os.path.exists(train_folder):
            train_folder = os.path.join(data_dir, 'Chula-ParasiteEgg-11', 'Chula-ParasiteEgg-11')
        
        X, y = self.load_images_from_folder(train_folder, label_prefix='train')
        
        if len(X) == 0:
            raise ValueError("Eğitim verileri bulunamadı!")
        
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"\nSınıf sayısı: {len(self.class_names)}")
        print(f"Sınıflar: {self.class_names}")
        
        unique, counts = np.unique(y_encoded, return_counts=True)
        print(f"Her sınıftan örnek sayısı:")
        for cls, count in zip(self.class_names[unique], counts):
            print(f"  {cls}: {count}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nEğitim seti: {X_train.shape[0]} örnek")
        print(f"Validasyon seti: {X_val.shape[0]} örnek")
        
        return X_train, X_val, y_train, y_val
    
    def prepare_test_data(self, data_dir='data'):
        """Test verilerini hazırlar"""
        # Önce data klasörü içinde ara
        test_folder = os.path.join(data_dir, '_test', 'test', 'data')
        
        if not os.path.exists(test_folder):
            # Sonra doğrudan _test klasöründe ara
            test_folder = os.path.join(data_dir, '_test', 'test')
            if not os.path.exists(test_folder):
                # Root dizinde ara
                test_folder = '_test/test/data'
                if not os.path.exists(test_folder):
                    test_folder = '_test/test'
        
        X_test, _ = self.load_images_from_folder(test_folder, label_prefix='')
        
        if len(X_test) == 0:
            raise ValueError(f"Test verileri bulunamadı! Aranan konumlar: {test_folder}")
        
        print(f"\nTest seti: {X_test.shape[0]} örnek")
        return X_test
    
    def get_data_loaders(self, X_train, X_val, y_train, y_val):
        """PyTorch DataLoader'ları oluşturur"""
        # Data augmentation
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ParasiteDataset(X_train, y_train, transform=train_transform)
        val_dataset = ParasiteDataset(X_val, y_val, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader

