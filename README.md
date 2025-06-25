# Transformer-Tabanli-Hayvan-Siniflandirma-Sistemi

Bu çalışma, MultiZoo veri seti
kullanılarak geliştirilen transformer tabanlı bir
görüntü sınıflandırma modelini kapsamaktadır.
Projenin amacı, bir görselde yer alan hayvan
türünü yüksek doğrulukla tahmin edebilen bir
sistem geliştirmektir. Vision Transformer modeli
ile eğitilen sistem, %97.9 eğitim doğruluğu ve
%94.7 doğrulama doğruluğu elde etmiştir. Ek
olarak, masaüstü bir arayüz tasarlanarak
kullanıcının test görseli yükleyip sonuç
alabilmesi sağlanmıştır.

YÖNTEM
Bu çalışmada, hayvan türlerini sınıflandırmaya
yönelik bir görüntü sınıflandırma modeli
geliştirilmiştir. Model olarak, görsel-ön işlem
adımlarının ardından Vision Transformer (ViT)
mimarisi kullanılmıştır. Modelin eğitimi PyTorch
kütüphanesiyle gerçekleştirilmiş, veri ön işleme ve
görselleştirme işlemleri için ise torchvision, sklearn,
pandas ve matplotlib kütüphanelerinden
yararlanılmıştır.
A. Donanım ve Cihaz Seçimi
Kodun başında Apple cihazlarında kullanılabilen
Metal Performance Shaders (MPS) kontrol edilmiş;
ancak çalıştırma sürecinde device = "cpu" olarak
sabitlenmiştir. Bu tercih, donanıma bağlı olarak
değiştirilebilmekle birlikte, çalışmanın tüm
aşamaları CPU üzerinde gerçekleştirilmiştir. Bu
sayede modelin donanımdan bağımsız olarak
çalışabilirliği test edilmiştir.


B. Veri Kümesi Yapısı
Kullanılan veri kümesi, MultiZooSplit adlı bir
klasör yapısı içerisinde train/ ve val/ alt klasörlerini
içerecek biçimde yapılandırılmıştır. Her bir alt
klasör, bir hayvan sınıfına ait görselleri
içermektedir. Toplamda 90 farklı hayvan sınıfı
tanımlanmış ve model bu 90 sınıf üzerinden
eğitilmiştir. ImageFolder yapısı kullanılarak veri
kümesi sınıf etiketleriyle otomatik olarak
eşleştirilmiştir.


C. Görüntü Ön İşleme (Data Augmentation)
Modelin aşırı öğrenmesini (overfitting) önlemek
amacıyla eğitim verilerine çeşitli görüntü işleme
dönüşümleri uygulanmıştır. Bu dönüşümler aşağıda
detaylandırılmıştır:
• Yeniden boyutlandırma: Tüm
görseller 224×224 piksele ölçeklendirilmiştir.
• Rastgele yatay çevirme: Görsellerin
%50 olasılıkla yatay olarak çevrilmesi sağlanmıştır.
• Renk bozulmaları (ColorJitter):
Parlaklık, kontrast ve doygunluk üzerinde küçük
rastgele değişiklikler uygulanmıştır.
• Normalize etme: Görseller RGB
kanal bazında ortalaması 0.5 ve standart sapması
0.5 olacak şekilde normalize edilmiştir.
Doğrulama verilerine ise sadece yeniden
boyutlandırma ve normalize işlemleri
uygulanmıştır. Böylece eğitim sırasında veri
çeşitliliği artırılırken, doğrulama aşamasında gerçek
veri dağılımı korunmuştur.



D. Model Mimarisi
Model olarak timm kütüphanesi üzerinden
vit_base_patch16_224 mimarisi kullanılmıştır. Bu
model, görselleri 16x16’lık parçalara (patch’lere)
bölerek her bir parçayı transformer yapısına dahil
eder. Modelin son sınıflandırma katmanı
(model.head) 90 sınıf sayısına uygun şekilde
aşağıdaki gibi güncellenmiştir.Modelin geri kalan
katmanları önceden eğitilmiş (pretrained)
ağırlıklarla başlatılmıştır. Bu, transfer öğrenmenin
bir örneği olup sınıflandırma doğruluğunu artırmak
amacıyla uygulanmıştır.


E. Eğitim Süreci
Modelin eğitimi toplam 5 epoch boyunca
gerçekleştirilmiştir. Her epoch’ta tüm eğitim verisi
bir kez işlenmiştir. Kullanılan optimizasyon
yöntemi AdamW olup, öğrenme oranı 1e-4 olarak
belirlenmiştir. Kayıp fonksiyonu olarak
CrossEntropyLoss kullanılmıştır. Eğitim sırasında
her batch için:
1. Görseller modele aktarılmış,
2. İleri yayılım (forward propagation)
yapılmış,
3. Kayıp değeri hesaplanmış,
4. Geri yayılım (backpropagation)
uygulanmış ve
5. Ağırlıklar güncellenmiştir.
Her epoch sonunda doğruluk ve ortalama kayıp
değerleri hesaplanmış ve kaydedilmiştir. Aşağıda 5
epoch boyunca elde edilen ortalama kayıp ve
doğruluk değerlerinin grafiği sunulmuştur



F. Doğrulama ve Değerlendirme
Eğitim tamamlandıktan sonra model değerlendirme
(eval) moduna geçirilmiştir. Doğrulama verisi
üzerinde model tahminleri yapılmış ve tahminler ile
gerçek etiketler karşılaştırılmıştır. Sınıflandırma
performansı, sklearn.metrics.classification_report
fonksiyonu ile ölçülmüştür. Elde edilen metrikler
bir CSV dosyasına kaydedilerek daha sonra analiz
edilebilecek biçimde dışa aktarılmıştır.



G. Model Kaydı ve Öğrenme Eğrisi
Eğitim sonunda elde edilen modelin ağırlıkları
vit_model_mps2.pth dosyası olarak kaydedilmiştir.
Ayrıca, epoch’lara göre değişen eğitim kaybı ve
doğruluk değerleri, learning_curve.png dosyasına
görsel olarak kaydedilmiştir. Bu dosya, modelin
öğrenme sürecini görselleştirmek ve eğitim
kalitesini değerlendirmek için kullanılmaktadır.
