"""
Parazit Yumurta Tespiti - Model Eğitimi
"""

import os
import argparse
from data_loader import ParasiteDataLoader
from model import create_cnn_model, create_transfer_learning_model, compile_model, get_model_summary
import tensorflow as tf
from tensorflow import keras
import numpy as np


def train_model(
    zip_path='Chula-ParasiteEgg-11.zip',
    use_transfer_learning=False,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    img_size=(224, 224),
    model_save_path='models/parasite_model.h5'
):
    """
    Modeli eğitir
    
    Args:
        zip_path: ZIP dosyası yolu
        use_transfer_learning: Transfer learning kullanılsın mı?
        epochs: Epoch sayısı
        batch_size: Batch boyutu
        learning_rate: Öğrenme oranı
        img_size: Görüntü boyutu
        model_save_path: Model kayıt yolu
    """
    print("=" * 60)
    print("PARAZİT YUMURTA TESPİTİ - MODEL EĞİTİMİ")
    print("=" * 60)
    
    # GPU kontrolü
    print("\nGPU Kontrolü:")
    print(f"TensorFlow versiyonu: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU bulundu: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("GPU bulunamadı, CPU kullanılacak")
    
    # Veri yükleyiciyi oluştur
    print("\n" + "=" * 60)
    print("VERİ YÜKLEME")
    print("=" * 60)
    data_loader = ParasiteDataLoader(zip_path, img_size=img_size, batch_size=batch_size)
    
    # ZIP dosyasını çıkar
    data_loader.extract_zip(extract_to='data')
    
    # Eğitim verilerini hazırla
    X_train, X_val, y_train, y_val = data_loader.prepare_training_data(data_dir='data')
    
    # Data generator'ları oluştur
    train_gen, val_gen = data_loader.get_data_generators(X_train, X_val, y_train, y_val)
    
    # Model oluştur
    print("\n" + "=" * 60)
    print("MODEL OLUŞTURMA")
    print("=" * 60)
    num_classes = len(data_loader.class_names)
    
    if use_transfer_learning:
        print("Transfer Learning modeli oluşturuluyor (MobileNetV2)...")
        model = create_transfer_learning_model(num_classes, img_size)
    else:
        print("CNN modeli oluşturuluyor...")
        model = create_cnn_model(num_classes, img_size)
    
    model = compile_model(model, learning_rate=learning_rate)
    get_model_summary(model)
    
    # Callback'ler
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger('training_history.csv', append=False)
    ]
    
    # Model eğitimi
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ")
    print("=" * 60)
    print(f"Epoch sayısı: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Öğrenme oranı: {learning_rate}")
    
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Model kaydet
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\nModel kaydedildi: {model_save_path}")
    
    # Sınıf isimlerini kaydet
    import pickle
    with open('models/class_names.pkl', 'wb') as f:
        pickle.dump(data_loader.class_names, f)
    print("Sınıf isimleri kaydedildi: models/class_names.pkl")
    
    # Eğitim geçmişini kaydet
    import json
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open('models/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("Eğitim geçmişi kaydedildi: models/training_history.json")
    
    print("\n" + "=" * 60)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 60)
    print(f"En iyi validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"En iyi validation loss: {min(history.history['val_loss']):.4f}")
    
    return model, history, data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parazit Yumurta Tespiti Model Eğitimi')
    parser.add_argument('--zip_path', type=str, default='Chula-ParasiteEgg-11.zip',
                        help='ZIP dosyası yolu')
    parser.add_argument('--transfer', action='store_true',
                        help='Transfer learning kullan')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch boyutu')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Öğrenme oranı')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Görüntü boyutu (224x224)')
    
    args = parser.parse_args()
    
    train_model(
        zip_path=args.zip_path,
        use_transfer_learning=args.transfer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_size=(args.img_size, args.img_size)
    )






