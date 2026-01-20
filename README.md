<img width="1089" height="768" alt="image" src="https://github.com/user-attachments/assets/cbfddfd9-21ab-40cc-9b4c-e5064df65bc9" />
<img width="968" height="658" alt="Ekran görüntüsü 2026-01-20 132009" src="https://github.com/user-attachments/assets/ce61ca35-b469-4b43-9610-d12c32a03067" />


# Akilli Konteyner doluluk Seviyesi tahmini

# Projenin Amacı

Bu projenin amacı, akıllı atık konteynerlerinden alınan sensör verilerini kullanarak konteynerlerin normal dolan mı yoksa hızlı dolan mı olduğunu tahmin eden bir makine öğrenmesi modeli geliştirmektir.

Çalışma boyunca konteyner türü, atık türü ve hacim sensörlerinden gelen veriler incelenmiş, bu veriler yardımıyla konteynerlerin dolma davranışları analiz edilmiştir. Amaç, hızlı dolma riski olan konteynerleri önceden tespit ederek daha verimli bir atık toplama süreci sağlamaktır.
# Hedef değişken

Projede hedef değişken olarak Hizli_Dolma kullanılmıştır. Bu değişken, konteynerin doluluk sensörlerinden elde edilen ölçümler yardımıyla oluşturulmuştur.

FL_A ve FL_B sensör değerleri arasındaki fark alınarak doluluk artışı hesaplanmıştır.

Hesaplanan doluluk artışlarının medyan değeri referans alınmıştır.

Medyanın altında kalan değerler 0 (normal dolma),

Medyanın üzerinde kalan değerler 1 (hızlı dolma) olarak etiketlenmiştir.
# Neler Yapıldı?
 Gerekli Kütüphaneler Eklendi
   
<img width="786" height="272" alt="image" src="https://github.com/user-attachments/assets/6435d1a9-cfa5-4c5e-81e6-42099ca61297" />

# Veri Seti 
df = pd.read_csv("Smart_Bin.csv")
Veri seti Smart_Bin.csv dosyasından okunmuştur. İlk olarak veri setinin genel yapısı incelenmiş, sütun bilgileri ve veri tipleri kontrol edilmiştir. Eksik veriler analiz edilmiş ve eksik veri sayısının az olması nedeniyle bu satırlar veri setinden çıkarılmıştır..
# Feature Engineerining

Konteynerlerin dolma hızını daha doğru temsil edebilmek için aşağıdaki özellik oluşturuldu:

df["Doluluk_Artisi"] = df["FL_B"] - df["FL_A"]


Bu değişken, konteynerin iki ölçüm arasındaki gerçek dolma hızını ifade etmektedir.
# Pivot Tablolama 

<img width="601" height="509" alt="image" src="https://github.com/user-attachments/assets/b1cd11d6-c8a9-4c8b-9aaa-ef86c499be5b" />

Konteyner türü ve atık türüne göre doluluk davranışlarını daha iyi anlayabilmek için pivot tablolar oluşturulmuştur. Bu tablolar sayesinde hangi konteyner türünün hangi atık türünde daha hızlı dolduğu açık bir şekilde görülmüştür.

Bu analizler, veri setindeki genel eğilimleri anlamak açısından önemli katkı sağlamıştır.



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



# Random Forest Modeli

Doluluk durumunu tahmin etmek için Random Forest modeli kullandım. Bu modeli seçmemin nedeni, birden fazla karar ağacıyla çalışarak verideki karmaşık ilişkileri daha iyi öğrenebilmesidir. Modeli eğitim verisiyle eğittim ve test verisi üzerinde denedim.

Model sonucunda yaklaşık %83 doğruluk elde ettim. Ayrıca modelin hangi özelliklere daha fazla önem verdiğini inceleyerek, doluluk tahmininde en etkili değişkenleri görselleştirdim.

<img width="813" height="276" alt="image" src="https://github.com/user-attachments/assets/4c7dea23-3f13-47e5-8f6b-fbf1954fc811" />


# KNN Modeli

Karşılaştırma yapmak için KNN modelini de denedim. KNN, benzer verileri birbirine yakın kabul ederek tahmin yapan bir yöntem olduğu için, modelden önce verileri ölçeklendirdim.

KNN modeliyle yaklaşık %81 doğruluk elde ettim. Ancak model, verideki mantıksal ilişkileri Random Forest kadar iyi yakalayamadığı için performansı biraz daha düşük kaldı.
# Genel Değerlendirme

Her iki model de konteyner doluluk durumunu tahmin etmede başarılı sonuçlar üretmiştir. Ancak:

Random Forest, karmaşık veri yapısını daha iyi öğrenmesi ve yorumlanabilirlik sunması nedeniyle daha başarılı olarak buldum .

KNN, basit ve sezgisel bir yaklaşım sunmasına rağmen, büyük veri setlerinde ve karmaşık ilişkilerde performans açısından sınırlı kaldı .

Bu nedenle proje kapsamında Random Forest modeli uygun model olarak tercih ettim.





