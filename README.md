Heart Attack Risk Prediction Model using Machine Learning
📌 Proje Tanımı
Bu proje, kalp krizi riskini tahmin etmek için makine öğrenmesi tekniklerini kullanmaktadır. Kaggle'dan alınan Cleveland Heart Disease veri seti üzerinde yapılan analizler sonucunda, çeşitli makine öğrenmesi algoritmalarıyla kalp hastalığı riski sınıflandırılmıştır.

🧠 Kullanılan Yöntemler
Veri Ön İşleme: Eksik değerlerin doldurulması, kategorik verilerin sayısallaştırılması ve verilerin normalize edilmesi işlemleri gerçekleştirilmiştir.
Özellik Seçimi: Modelin performansını artırmak için önemli özellikler seçilmiştir.
Modelleme: Karar Ağaçları (Decision Trees), Destek Vektör Makineleri (SVM), K-En Yakın Komşu (KNN) ve Naive Bayes gibi çeşitli makine öğrenmesi algoritmaları kullanılmıştır.
Model Değerlendirme: Modelin doğruluk, hassasiyet, özgüllük ve F1 skoru gibi metriklerle performansı değerlendirilmiştir.
🧪 Veri Seti
Veri seti, Cleveland Heart Disease veri setinden alınmıştır ve aşağıdaki özellikleri içermektedir:
Yaş: Hastanın yaşı
Cinsiyet: 0 = Kadın, 1 = Erkek
Göğüs Ağrısı Türü: 0 = Asimptomatik, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Typical Angina
Kan Basıncı: Sistolik ve diyastolik kan basıncı değerleri
Kolesterol: Serum kolesterol seviyesi
Kan Şekeri: Açlık kan şekeri seviyesi
EKG Sonuçları: Elektrokardiyogram sonuçları
Maksimum Kalp Hızı: Maksimum kalp atış hızı
Egzersiz Anginası: 0 = Hayır, 1 = Evet
ST Depresyonu: ST depresyonu değeri
Tepe ST Segmenti: Tepe ST segmenti eğimi
Major Damarlar: Görüntülemede görünen ana damar sayısı
Talasemi: 0 = Normal, 1 = Taşınan, 2 = Orta, 3 = Şiddetli
🚀 Kurulum ve Çalıştırma
Gerekli Kütüphanelerin Kurulumu:
pip install numpy pandas matplotlib seaborn scikit-learn
Jupyter Notebook ile Çalıştırma:
jupyter notebook
Ardından, Heart Attack Analysis.ipynb dosyasını açarak adım adım analizleri inceleyebilirsiniz.
📈 Sonuçlar
Modelin doğruluk oranı, kullanılan algoritmaya bağlı olarak değişiklik göstermektedir. En yüksek doğruluk oranı, XGBoost algoritması ile elde edilmiştir. Sonuçlar, kalp hastalığı riskini tahmin etmede makine öğrenmesi tekniklerinin etkinliğini göstermektedir.
