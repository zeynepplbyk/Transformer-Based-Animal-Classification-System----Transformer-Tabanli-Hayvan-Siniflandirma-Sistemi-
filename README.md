## 🐾 Transformer-Tabanlı Hayvan Sınıflandırma Sistemi

Bu proje, MultiZoo veri seti kullanılarak geliştirilen Vision Transformer (ViT) tabanlı bir görüntü sınıflandırma modelini kapsamaktadır.
Amaç, bir görseldeki hayvan türünü yüksek doğrulukla tahmin edebilen bir sistem geliştirmektir.

# 📌 Sonuçlar:
	•	✅ Eğitim Doğruluğu: %97.9
	•	✅ Doğrulama Doğruluğu: %94.7
	•	✅ Masaüstü arayüzü ile kullanıcı dostu test imkanı

# Kullanılan Teknolojiler
	•	PyTorch (model eğitimi)
	•	timm (Vision Transformer modelleri)
	•	torchvision, sklearn, pandas, matplotlib (ön işleme ve görselleştirme)

# 🛠 Yöntem

1. Görüntü Ön İşleme (Data Augmentation)
	•	📏 Yeniden boyutlandırma (224×224)
	•	🔄 Rastgele yatay çevirme (%50)
	•	🎨 Renk bozulmaları (ColorJitter)
	•	⚖️ Normalize etme (mean=0.5, std=0.5)

2. Model Mimarisi
	•	Model: vit_base_patch16_224 (timm üzerinden)
	•	Transfer Learning: Önceden eğitilmiş ağırlıklar kullanıldı
	•	Çıkış katmanı: 90 sınıf için özelleştirildi

3. Eğitim Süreci
	•	Epoch: 5
	•	Optimizer: AdamW (lr=1e-4)
	•	Loss: CrossEntropyLoss
	•	Cihaz: CPU (MPS kontrol edildi, device=“cpu” sabitlendi)


# 📊 Sonuçlar
Eğitim Doğruluğu: %97.9
Doğrulama Doğruluğu: %94.7

# Model Kaydı
	•	Eğitilen model: vit_model_mps2.pth
	•	Öğrenme eğrisi: learning_curve.png


# Masaüstü Arayüz
Kullanıcı, uygulama arayüzü üzerinden test görselleri yükleyip tahmin sonuçlarını görebilmektedir.


# Repository'yi klonla
git clone https://github.com/kullaniciadi/Transformer-Tabanli-Hayvan-Siniflandirma-Sistemi.git
cd Transformer-Tabanli-Hayvan-Siniflandirma-Sistemi

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Modeli çalıştır
python train.py
python evaluate.py

