## 📌 Proje Tanımı

Bu proje, kalp krizi riskini tahmin etmek için makine öğrenmesi tekniklerini kullanmaktadır. Kaggle'dan alınan Cleveland Heart Disease veri seti üzerinde yapılan analizler sonucunda, çeşitli makine öğrenmesi algoritmalarıyla kalp hastalığı riski sınıflandırılmıştır.

## 🧠 Kullanılan Yöntemler

- **Veri Ön İşleme:** Eksik değerlerin doldurulması, kategorik verilerin sayısallaştırılması ve verilerin normalize edilmesi işlemleri gerçekleştirilmiştir.
- **Özellik Seçimi:** Modelin performansını artırmak için önemli özellikler seçilmiştir.
- **Modelleme:** Karar Ağaçları (Decision Trees), Destek Vektör Makineleri (SVM), K-En Yakın Komşu (KNN) ve Naive Bayes gibi çeşitli makine öğrenmesi algoritmaları kullanılmıştır.
- **Model Değerlendirme:** Modelin doğruluk, hassasiyet, özgüllük ve F1 skoru gibi metriklerle performansı değerlendirilmiştir.

## 🧪 Veri Seti

Veri seti, Cleveland Heart Disease veri setinden alınmıştır ve aşağıdaki özellikleri içermektedir:

- **Yaş:** Hastanın yaşı
- **Cinsiyet:** 0 = Kadın, 1 = Erkek
- **Göğüs Ağrısı Türü:** 0 = Asimptomatik, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Typical Angina
- **Kan Basıncı:** Sistolik ve diyastolik kan basıncı değerleri
- **Kolesterol:** Serum kolesterol seviyesi
- **Kan Şekeri:** Açlık kan şekeri seviyesi
- **EKG Sonuçları:** Elektrokardiyogram sonuçları
- **Maksimum Kalp Hızı:** Maksimum kalp atış hızı
- **Egzersiz Anginası:** 0 = Hayır, 1 = Evet
- **ST Depresyonu:** ST depresyonu değeri
- **Tepe ST Segmenti:** Tepe ST segmenti eğimi
- **Major Damarlar:** Görüntülemede görünen ana damar sayısı
- **Talasemi:** 0 = Normal, 1 = Taşınan, 2 = Orta, 3 = Şiddetli

## 🧪 Veri Seti

Bu projede, kalp hastalığı riskini tahmin etmek için [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) veri seti kullanılmıştır. Bu veri seti, 2022 yılına ait CDC (Centers for Disease Control and Prevention) anket verilerini içermektedir ve aşağıdaki özellikleri sunmaktadır:

- **Özellikler:** 18 sütun (9 boolean, 5 string, 4 sayısal)
- **Örnek Sayısı:** 320.000+ satır
- **İçerik:** Katılımcıların kalp hastalığına dair kişisel sağlık göstergeleri

Veri seti, kalp hastalığı riskini etkileyebilecek çeşitli faktörleri analiz etmek için kapsamlı bir kaynak sunmaktadır.


## 🚀 Kurulum ve Çalıştırma

1. **Gerekli Kütüphanelerin Kurulumu:**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn

