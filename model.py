"""
Parazit Yumurta Tespiti - Model Mimarisi
CNN tabanlı derin öğrenme modeli
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def create_cnn_model(num_classes, img_size=(224, 224)):
    """
    CNN modeli oluşturur
    
    Args:
        num_classes: Sınıf sayısı (parazit türü sayısı)
        img_size: Görüntü boyutu (yükseklik, genişlik)
    
    Returns:
        model: Keras modeli
    """
    model = models.Sequential([
        # İlk Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # İkinci Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Üçüncü Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Dördüncü Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Flatten ve Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_transfer_learning_model(num_classes, img_size=(224, 224)):
    """
    Transfer Learning ile model oluşturur (MobileNetV2 kullanarak)
    
    Args:
        num_classes: Sınıf sayısı
        img_size: Görüntü boyutu
    
    Returns:
        model: Keras modeli
    """
    # Pre-trained MobileNetV2 modelini yükle
    base_model = keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Base model katmanlarını dondur (opsiyonel - fine-tuning için)
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Modeli derler
    
    Args:
        model: Keras modeli
        learning_rate: Öğrenme oranı
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model):
    """Model özetini yazdırır"""
    model.summary()
    
    # Model parametre sayısını hesapla
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nToplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    print(f"Eğitilemez parametre sayısı: {non_trainable_params:,}")






