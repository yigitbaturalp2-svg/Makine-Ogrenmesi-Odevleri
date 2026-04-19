# Uzak Bir Galaksinin Parlaklık Analizi (MCMC)

Bu proje, bir gök cisminin gerçek parlaklığını ve veri setindeki belirsizliği (standart sapma) gürültülü gözlem verilerini kullanarak Bayesyen yöntemlerle tahmin eden bir simülasyon çalışmasıdır.

## 1. Problem Tanımı
Astronomi projelerinde gözlemler genellikle atmosferik dalgalanmalar, teleskop gürültüsü ve kozmik toz gibi nedenlerle gürültülüdür. Bu projede, bilinen "gerçek" değerlerden üretilen sentetik bir veri seti üzerinden, Bayesyen çıkarımın (Bayesian Inference) bu belirsizlikleri nasıl yönettiği incelenmiştir.

## 2. Veri Seti
- **Gerçek Parlaklık ($\mu$):** 150.0
- **Gözlem Hatası ($\sigma$):** 10.0
- **Gözlem Sayısı ($n$):** 50
- **Rastgelelik:** `np.random.seed(42)` kullanılarak üretilmiştir.

## 3. Yöntem
Analizde **MCMC (Markov Chain Monte Carlo)** algoritması, özellikle `emcee` kütüphanesi kullanılarak uygulanmıştır.
- **Log-Likelihood:** Verilerin modele olan uygunluğu hesaplanmıştır.
- **Log-Prior:** Parametreler için geniş ve informatif olmayan (uninformative) sınırlar belirlenmiştir ($\mu \in [0, 300]$, $\sigma \in [0, 50]$).
- **Log-Posterior:** Prior ve Likelihood birleştirilerek hedef fonksiyon oluşturulmuştur.

## 4. Sonuçlar

### Parametre Karşılaştırma Tablosu

| Değişken | Gerçek Değer (Girdi) | Tahmin Edilen (Median) | Alt Sınır (%16) | Üst Sınır (%84) | Mutlak Hata |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\mu$ (Parlaklık) | 150.0 | 147.79 | 146.43 | 149.07 | 2.21 |
| $\sigma$ (Hata Payı) | 10.0 | 9.49 | 8.55 | 10.53 | 0.51 |

## 5. Görselleştirme
Elde edilen posterior dağılımları ve parametreler arasındaki korelasyonu gösteren **Corner Plot** grafiği `corner_plot.png` dosyasında yer almaktadır.

## 6. Bilimsel Yorum ve Tartışma
- **Doğruluk (Accuracy):** Model, %6-7 gürültü oranına rağmen gerçek parlaklık değerine %1.5 hata payı ile yaklaşmıştır.
- **Hassasiyet (Precision):** Ortalamanın ($\mu$) tahmini, varyansın ($\sigma$) tahminine göre daha dar bir güven aralığına sahiptir. Bunun nedeni 50 gözlemin merkezi eğilimi belirlemede yeterli olması, ancak varyansı kesinleştirmek için daha fazla veriye ihtiyaç duyulmasıdır.
- **Korelasyon:** Corner plot üzerindeki elipsin dikey duruşu, $\mu$ ve $\sigma$ parametrelerinin istatistiksel olarak birbirinden bağımsız olduğunu doğrulamaktadır.

## 7. Ek Analizler (Senaryo Soruları)

### 7.1. Prior Etkisi
**Soru:** Eğer parlaklık için çok dar bir prior seçseydiniz (örneğin 100-110 arası), sonuç nasıl değişirdi?
**Yanıt:** Bayesyen çıkarımda "Prior", veriyi görmeden önceki bilgimizdir. Eğer mu için [100, 110] gibi dar ve gerçek değerden (150) uzak bir aralık seçilirse, model "gerçek" değeri (150) asla bulamazdı. Posterior dağılımı 110 sınırına yığılırdı. Bu durum, yanlış bilgilendirilmiş (misinformed) bir prior'ın veriyi nasıl domine edebileceğini ve yanlı sonuçlara yol açabileceğini gösterir.

### 7.2. Veri Miktarı Etkisi
**Soru:** Gözlem sayısını (n_obs) 5'e düşürdüğünüzde posterior dağılımının genişliği (belirsizlik) nasıl etkileniyor?
**Yanıt:** Gözlem sayısı $n$ azaldığında, istatistiksel belirsizlik artar. 5 gözlem durumunda posterior dağılımları (Corner Plot'taki tepeler) çok daha geniş ve yayvan olurdu. Bu, "Precision" (hassasiyet) kaybı demektir. Bayesyen yaklaşımda veri azaldıkça "Prior"ın etkisi artar; veri çoğaldıkça ise veri (Likelihood) prior'ı domine eder.
