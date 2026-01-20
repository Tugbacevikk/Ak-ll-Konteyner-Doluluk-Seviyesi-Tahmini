# Akıll-Konteyner-Doluluk-Seviyesi-Tahmini

# Projenin Amacı

Bu projenin amacı, akıllı atık konteynerlerinden elde edilen sensör verileri kullanılarak konteynerlerin doluluk durumunun normal mi yoksa hızlı dolan bir durumda mı olduğunu tahmin eden bir makine öğrenmesi modeli geliştirmektir.

Çalışma kapsamında konteyner türü, atık türü ve hacim sensörü verileri analiz edilerek, doluluk davranışları incelenmiş ve hızlı dolma riski taşıyan durumların önceden tespit edilmesi hedeflenmiştir.

# Target (Hedef Değişken)

Hedef değişken Hizli_Dolma, konteyner doluluk seviyesini temsil eden sensör ölçümleri üzerinden oluşturulmuştur.

FL_A ve FL_B sensör değerleri arasındaki fark alınarak doluluk artışı hesaplanmıştır.

Doluluk artışının medyan değeri referans alınmıştır.

Medyanın:

altında kalan değerler 0 (normal dolma)

üzerinde kalan değerler 1 (hızlı dolma)
olarak etiketlenmiştir.

Bu sayede problem ikili sınıflandırma (binary classification) problemine dönüştürülmüştür.
Neler Yapıldı?
 Gerekli Kütüphaneler Eklendi
   
<img width="786" height="272" alt="image" src="https://github.com/user-attachments/assets/6435d1a9-cfa5-4c5e-81e6-42099ca61297" />

# Veri Seti Alındı ve İncelendi
df = pd.read_csv("Smart_Bin.csv")
df.head()
df.info()

Veri seti genel yapısı incelendi.

Eksik veriler kontrol edildi.

Eksik verilerin az olması nedeniyle ilgili satırlar veri setinden çıkarıldı.
# Feature Engineering (Doluluk Artışı)

Konteynerlerin dolma hızını daha doğru temsil edebilmek için aşağıdaki özellik oluşturuldu:

df["Doluluk_Artisi"] = df["FL_B"] - df["FL_A"]


Bu değişken, konteynerin iki ölçüm arasındaki gerçek dolma hızını ifade etmektedir.
# Pivot Tablolama (Davranış Analizi)

Zamansal veya kategorik bilgileri doğrudan kullanmak yerine, bu bilgilerin doluluk üzerindeki etkisini yansıtan pivot tablolar oluşturulmuştur.


<img width="601" height="509" alt="image" src="https://github.com/user-attachments/assets/b1cd11d6-c8a9-4c8b-9aaa-ef86c499be5b" />


Bu kodlar hangi konteyner türünün hangi atık türünde daha yüksek doluluk seviyelerine ulaştığını göstermektedir ve hangi konteyner ve atık türü kombinasyonunun daha hızlı dolduğunu ortaya koymaktadır.

# Görselleştirme

Ortalama doluluk değerleri için ısı haritası oluşturulmuştur.

Dolma hızlarını karşılaştırmak için ayrı bir ısı haritası çizilmiştir.

Konteyner türlerine göre ortalama doluluk değerleri bar grafik ile gösterilmiştir.

Bu görseller, veri setindeki temel doluluk davranışlarının daha kolay yorumlanmasını sağlamıştır.

<img width="1914" height="1021" alt="Ekran görüntüsü 2026-01-20 123959" src="https://github.com/user-attachments/assets/2e78b2a5-4be8-4ac0-bc0f-02e88f1d7884" />

<img width="1909" height="972" alt="Ekran görüntüsü 2026-01-20 124016" src="https://github.com/user-attachments/assets/c4631728-830c-4472-a02d-d1b802804ee9" />

<img width="1919" height="973" alt="Ekran görüntüsü 2026-01-20 124035" src="https://github.com/user-attachments/assets/a7ce6cc4-c09f-44a4-869f-34551c6f15d8" />

<img width="1918" height="967" alt="Ekran görüntüsü 2026-01-20 124052" src="https://github.com/user-attachments/assets/06d98ec0-17b3-4c76-996e-a041a7f459a7" />

# Modelleme İçin Veri Hazırlığı

Container Type ve Recyclable fraction sütunları Label Encoding ile sayısallaştırılmıştır.

Model girdileri olarak:

Konteyner türü

Atık türü

Hacim sensörü (VS)
kullanılmıştır.

X = df[["C_Encoded", "W_Encoded", "VS"]]
y = df["Hizli_Dolma"]


Veri seti %80 eğitim, %20 test olacak şekilde ayrılmıştır.

# Random Forest Modeli

Doluluk durumunu tahmin etmek için Random Forest modeli kullandım. Bu modeli seçmemin nedeni, birden fazla karar ağacıyla çalışarak verideki karmaşık ilişkileri daha iyi öğrenebilmesidir. Modeli eğitim verisiyle eğittim ve test verisi üzerinde denedim.

Model sonucunda yaklaşık %83 doğruluk elde ettim. Ayrıca modelin hangi özelliklere daha fazla önem verdiğini inceleyerek, doluluk tahmininde en etkili değişkenleri görselleştirdim.

<img width="813" height="276" alt="image" src="https://github.com/user-attachments/assets/4c7dea23-3f13-47e5-8f6b-fbf1954fc811" />


# KNN Modeli

Karşılaştırma yapmak için KNN modelini de denedim. KNN, benzer verileri birbirine yakın kabul ederek tahmin yapan bir yöntem olduğu için, modelden önce verileri ölçeklendirdim.

KNN modeliyle yaklaşık %81 doğruluk elde ettim. Ancak model, verideki mantıksal ilişkileri Random Forest kadar iyi yakalayamadığı için performansı biraz daha düşük kaldı.
# Genel Değerlendirme

Her iki model de konteyner doluluk durumunu tahmin etmede başarılı sonuçlar üretmiştir. Ancak:

Random Forest, karmaşık veri yapısını daha iyi öğrenmesi ve yorumlanabilirlik (feature importance) sunması nedeniyle daha başarılı bulunmuştur.

KNN, basit ve sezgisel bir yaklaşım sunmasına rağmen, büyük veri setlerinde ve karmaşık ilişkilerde performans açısından sınırlı kalmıştır.

Bu nedenle proje kapsamında Random Forest modeli uygun model olarak tercih edilmiştir.





