# YZM212 Makine Öğrenmesi - 2. Laboratuvar Ödevi

## MLE ile Akıllı Şehir Planlaması (Trafik Yoğunluğu Analizi)

Bu ödevde bir ana caddeden bir dakikada geçen araç sayılarını Poisson dağılımı ile modelledim. Parametre olan $\lambda$ değerini hem teorik türetimle hem de Python ile sayısal olarak hesapladım.

## Problem

Amaç, eldeki gözlemlerden caddenin tipik trafik seviyesini temsil eden Poisson parametresini bulmaktır.

## Veri

Kullanılan gözlemler:

- [12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]

Bu veri, seçilen zaman dilimlerinde bir dakikada geçen araç sayılarını ifade etmektedir.

## Yöntem

1. Poisson PMF üzerinden likelihood ve log-likelihood ifadelerini yazdım.
2. Teorik olarak Poisson için MLE'nin $\hat{\lambda}=\bar{k}$ olduğunu gösterdim.
3. Negatif log-likelihood fonksiyonunu kurup `scipy.optimize.minimize` ile sayısal çözüm aldım.
4. PMF ve histogramı aynı grafikte karşılaştırdım.
5. Veriye 200 değerini ekleyip outlier etkisini ayrıca inceledim.

## Sonuçlar

- Analitik MLE: $\hat{\lambda}_{ana} \approx 12.142857$
- Sayısal MLE: $\hat{\lambda}_{num} \approx 12.142862$
- İki sonuç arasındaki fark çok küçüktür.

Outlier eklendikten sonra:

- Yeni MLE belirgin biçimde artmıştır.
- Artış oranı yaklaşık %103 seviyesine ulaşmıştır.

Bu durum, ortalamaya dayalı yöntemlerin aykırı değerlerden hızlı etkilendiğini açıkça göstermektedir.

## Yorum / Tartışma

Model, verinin yoğun olduğu aralıkta genel olarak iyi bir uyum veriyor. Yine de tek bir hatalı kayıt bile parametreyi ciddi şekilde değiştirebildiği için şu önlemler gereklidir:

1. Veri girişinde temel aralık kontrolleri yapılmalıdır.
2. Aykırı değerler raporlanıp temizleme kuralı uygulanmalıdır.
3. Uzun dönemli veriyle model periyodik olarak güncellenmelidir.

## Proje Dosyaları

- `homework2_mle_ozgun_taslak.ipynb`: Notebook çözüm dosyası
- `RAPOR.pdf`: Rapor dosyası
- `figures/`: Grafik çıktıları

## Kullanılan Kütüphaneler

- numpy
- scipy
- matplotlib

