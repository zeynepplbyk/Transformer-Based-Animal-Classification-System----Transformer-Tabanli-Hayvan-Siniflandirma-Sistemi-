import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import os

class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
               'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog',
               'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
               'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
               'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
               'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse',
               'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot',
               'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat',
               'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake',
               'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale',
               'wolf', 'wombat', 'woodpecker', 'zebra']


model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(class_names))
model.load_state_dict(torch.load("vit_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor).logits
        probabilities = F.softmax(logits, dim=1)
    top_prob, top_idx = torch.max(probabilities, dim=1)
    result = (class_names[top_idx.item()], top_prob.item() * 100)
    return result, image

def predict_folder(folder_path):
    all_images = [os.path.join(root, file)
                  for root, _, files in os.walk(folder_path)
                  for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    correct = 0
    total = 0
    for img_path in all_images:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor).logits
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
        predicted_label = class_names[pred_idx]
        true_label = os.path.basename(os.path.dirname(img_path))
        if true_label == predicted_label:
            correct += 1
        total += 1
    accuracy = (correct / total) * 100 if total else 0
    return accuracy, correct, total

class AnimalClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BiHayvan | Görselden Hayvan Tanıma")
        self.geometry("600x700")
        self.configure(bg="#1e1e2f")
        self.image_path = None
        
        

        self.header = Label(self, text="🐾 BiHayvan", font=("Segoe UI", 28, "bold"), fg="#ff69b4", bg="#1e1e2f")
        self.header.pack(pady=(20, 5))

        self.subheader = Label(self, text="Vision Transformer ile Hayvan Tanıma", font=("Segoe UI", 14), fg="#bbbbbb", bg="#1e1e2f")
        self.subheader.pack(pady=(0, 20))

        self.image_label = Label(self, text="Henüz görsel yüklenmedi", width=40, height=15,
                                  bg="#2a2a3d", fg="#888888", font=("Segoe UI", 12), relief="groove")
        self.image_label.pack(pady=15)

        btn_style = {"font": ("Segoe UI", 12, "bold"), "bg": "#ff69b4", "fg": "black", "padx": 12, "pady": 8}

        btn_frame = Frame(self, bg="#1e1e2f")
        btn_frame.pack(pady=10)

        self.load_button = Button(btn_frame, text="📁 Resim Seç", command=self.load_image, **btn_style)
        self.load_button.pack(side="left", padx=10)

        self.predict_button = Button(btn_frame, text="🔎 Tahmin Et", command=self.predict, **btn_style)
        self.predict_button.pack(side="left", padx=10)

        self.result_label = Label(self, text="", font=("Segoe UI", 14), fg="#00ffcc", bg="#1e1e2f", wraplength=550, justify="center")
        self.result_label.pack(pady=15)

        self.batch_title = Label(self, text="📂 Klasör ile Toplu Test", font=("Segoe UI", 18, "bold"), fg="#ffaa00", bg="#1e1e2f")
        self.batch_title.pack(pady=(30, 10))

        self.batch_button = Button(self, text="📂 Klasör Seç ve Test Et", command=self.load_folder, font=("Segoe UI", 12, "bold"), bg="#ff8c00", fg="black", padx=15, pady=8)
        self.batch_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path).resize((320, 320))
            img_tk = ImageTk.PhotoImage(image)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk
            self.result_label.config(text="")

    def predict(self):
        if not self.image_path:
            return
        result, _ = predict_image(self.image_path)
        name, prob = result
        result_text = f"\n\nTahmin Edilen Hayvan: {name}\nGüven: %{prob:.2f}"
        self.result_label.config(text=result_text)

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            accuracy, correct, total = predict_folder(folder_path)
            self.result_label.config(
                text=f"\n📊 Toplu Test Sonucu:\n✅ Doğruluk: %{accuracy:.2f} ({correct}/{total})",
                fg="#ffd700"
            )

if __name__ == "__main__":
    app = AnimalClassifierApp()
    app.mainloop()