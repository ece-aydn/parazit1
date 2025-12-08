"""
Parazit Yumurta Tespiti - Veri Yükleme Modülü
Bu modül, Chula-ParasiteEgg-11 veri setini yükler ve işler.
"""

import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


class ParasiteDataLoader:
    """Parazit görüntü verilerini yükleyen ve işleyen sınıf"""
    
    def __init__(self, zip_path, img_size=(224, 224), batch_size=32):
        """
        Args:
            zip_path: ZIP dosyasının yolu
            img_size: Görüntülerin yeniden boyutlandırılacağı boyut (yükseklik, genişlik)
            batch_size: Batch boyutu
        """
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
        """
        Klasörden görüntüleri ve etiketlerini yükler
        
        Args:
            folder_path: Görüntülerin bulunduğu klasör yolu
            label_prefix: Etiket öneki (parazit türü için)
        
        Returns:
            images: Görüntü dizisi
            labels: Etiket dizisi
        """
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"Klasör bulunamadı: {folder_path}")
            return np.array(images), np.array(labels)
        
        # Ana klasörden görüntüleri yükle
        data_path = os.path.join(folder_path, 'data')
        if not os.path.exists(data_path):
            data_path = folder_path
        
        print(f"Görüntüler yükleniyor: {data_path}")
        
        for filename in os.listdir(data_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(data_path, filename)
                
                try:
                    # Görüntüyü yükle ve yeniden boyutlandır
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0  # Normalize et [0, 1]
                    
                    images.append(img_array)
                    
                    # Etiketi çıkar
                    if label_prefix:
                        # Ana klasör: "Parazit_Türü_0001.jpg" formatından türü çıkar
                        label = filename.rsplit('_', 1)[0]  # Son _ işaretinden öncesini al
                        labels.append(label)
                    else:
                        # Test klasörü: sadece numara var, etiket yok
                        labels.append('unknown')
                        
                except Exception as e:
                    print(f"Hata: {img_path} yüklenemedi - {e}")
                    continue
        
        print(f"Toplam {len(images)} görüntü yüklendi")
        return np.array(images), np.array(labels)
    
    def prepare_training_data(self, data_dir='data'):
        """
        Eğitim verilerini hazırlar
        
        Returns:
            X_train, X_val, y_train, y_val: Eğitim ve validasyon verileri
        """
        # Ana klasörden eğitim verilerini yükle
        train_folder = os.path.join(data_dir, 'Chula-ParasiteEgg-11', 'Chula-ParasiteEgg-11', 'data')
        
        if not os.path.exists(train_folder):
            # Alternatif yol
            train_folder = os.path.join(data_dir, 'Chula-ParasiteEgg-11', 'Chula-ParasiteEgg-11')
        
        X, y = self.load_images_from_folder(train_folder, label_prefix='train')
        
        if len(X) == 0:
            raise ValueError("Eğitim verileri bulunamadı!")
        
        # Etiketleri encode et
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"\nSınıf sayısı: {len(self.class_names)}")
        print(f"Sınıflar: {self.class_names}")
        print(f"Her sınıftan örnek sayısı:")
        unique, counts = np.unique(y_encoded, return_counts=True)
        for cls, count in zip(self.class_names[unique], counts):
            print(f"  {cls}: {count}")
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nEğitim seti: {X_train.shape[0]} örnek")
        print(f"Validasyon seti: {X_val.shape[0]} örnek")
        
        return X_train, X_val, y_train, y_val
    
    def prepare_test_data(self, data_dir='data'):
        """
        Test verilerini hazırlar
        
        Returns:
            X_test: Test görüntüleri
        """
        # Test klasöründen verileri yükle
        test_folder = os.path.join(data_dir, '_test', 'test', 'data')
        
        if not os.path.exists(test_folder):
            test_folder = os.path.join(data_dir, '_test', 'test')
        
        X_test, _ = self.load_images_from_folder(test_folder, label_prefix='')
        
        if len(X_test) == 0:
            raise ValueError("Test verileri bulunamadı!")
        
        print(f"\nTest seti: {X_test.shape[0]} örnek")
        
        return X_test
    
    def get_data_generators(self, X_train, X_val, y_train, y_val):
        """
        Data augmentation ile data generator'lar oluşturur
        
        Returns:
            train_gen, val_gen: Eğitim ve validasyon generator'ları
        """
        # Data augmentation için ImageDataGenerator
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator()
        
        # Generator'ları oluştur
        train_gen = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_gen = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_gen, val_gen






