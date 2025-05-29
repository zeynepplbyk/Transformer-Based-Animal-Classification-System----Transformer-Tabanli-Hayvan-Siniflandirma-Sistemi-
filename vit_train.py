import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Donanım kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 2. Yol ve parametreler
data_dir = "/Users/zeynep/Desktop/yazlab_2.3/train"
batch_size = 8
img_size = 224
val_ratio = 0.2

# 3. Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 4. Dataset yükleme ve ayırma
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = full_dataset.classes
train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# 5. DataLoader'lar
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 6. ViT modeli
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(class_names)
)
model.to(device)

# 7. Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 8. Eğitim döngüsü
def train_model(model, train_loader, val_loader, epochs=5):
    precision_list = []
    recall_list = []
    f1_list = []
    print("Eğitim başlatıldı...")
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} başlıyor...")
        model.train()
        total, correct = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"  → Batch {batch_idx + 1} işleniyor...")
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # İlk batch'ten sonra takılma yoksa devam
            if batch_idx == 0:
                print("  ✅ İlk batch başarıyla işlendi.")

        train_acc = correct / total
        train_acc_history.append(train_acc)

        # Doğrulama + metrikler
        model.eval()
        val_total, val_correct = 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_acc_history.append(val_acc)

        # Metrikler
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        print(f"📊 Epoch [{epoch+1}/{epochs}] sonuçları:")
        print(f"  Train Accuracy:     {train_acc:.4f}")
        print(f"  Validation Accuracy:{val_acc:.4f}")
        print(f"  Precision (macro):  {precision:.4f}")
        print(f"  Recall (macro):     {recall:.4f}")
        print(f"  F1 Score (macro):   {f1:.4f}")

    return train_acc_history, val_acc_history, precision_list, recall_list, f1_list

# 9. Eğitimi başlat
train_acc, val_acc, precision_list, recall_list, f1_list = train_model(model, train_loader, val_loader)

# 10. Grafik çiz
epochs_range = range(1, len(train_acc) + 1)

# Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.close()

# Precision Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, precision_list, label='Precision (macro)', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision')
plt.legend()
plt.grid(True)
plt.savefig("precision_plot.png")
plt.close()

# Recall Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, recall_list, label='Recall (macro)', color='green')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Validation Recall')
plt.legend()
plt.grid(True)
plt.savefig("recall_plot.png")
plt.close()

# F1 Score Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, f1_list, label='F1 Score (macro)', color='red')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Validation F1 Score')
plt.legend()
plt.grid(True)
plt.savefig("f1_score_plot.png")
plt.close()

print("✅ Tüm metrik grafik dosyaları kaydedildi.")

# 11. Modeli kaydet
torch.save(model.state_dict(), "vit_model.pth")
print("✅ Model kaydedildi: vit_model.pth")