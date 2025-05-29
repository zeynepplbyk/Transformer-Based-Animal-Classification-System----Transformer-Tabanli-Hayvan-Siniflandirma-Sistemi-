from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

# Veri yolu
data_dir = "/Users/zeynep/Desktop/yazlab_2.3/train"

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset yükleniyor
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Eğitim ve validation oranları
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Sabit seed ile bölme
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# DataLoader'lar
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Bilgi yazdır
print(f"Toplam Görüntü: {len(full_dataset)}")
print(f"Eğitim Seti: {len(train_dataset)} görüntü")
print(f"Doğrulama Seti: {len(val_dataset)} görüntü")
print(f"Sınıf Sayısı: {len(full_dataset.classes)}")
print(f"Sınıf İsimleri: {full_dataset.classes}")