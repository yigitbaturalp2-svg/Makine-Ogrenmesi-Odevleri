# YZM212 Makine Ogrenmesi - 2. Laboratuvar Odevi

## MLE ile Akilli Sehir Planlamasi (Trafik Yogunlugu Analizi)

Bu odevde bir ana caddeden bir dakikada gecen arac sayilarini Poisson dagilimi ile modelledim. Parametre olan $\lambda$ degerini hem teorik turetimle hem de Python ile sayisal olarak hesapladim.

## Problem

Amaç, eldeki gozlemlerden caddenin tipik trafik seviyesini temsil eden Poisson parametresini bulmaktir.

## Veri

Kullanilan gozlemler:

- [12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]

Bu veri, secilen zaman dilimlerinde bir dakikada gecen arac sayilarini ifade etmektedir.

## Yontem

1. Poisson PMF uzerinden likelihood ve log-likelihood ifadelerini yazdim.
2. Teorik olarak Poisson icin MLE'nin $\hat{\lambda}=\bar{k}$ oldugunu gosterdim.
3. Negatif log-likelihood fonksiyonunu kurup `scipy.optimize.minimize` ile sayisal cozum aldim.
4. PMF ve histogrami ayni grafikte karsilastirdim.
5. Veriye 200 degerini ekleyip outlier etkisini ayrica inceledim.

## Sonuclar

- Analitik MLE: $\hat{\lambda}_{ana} \approx 12.142857$
- Sayisal MLE: $\hat{\lambda}_{num} \approx 12.142862$
- Iki sonuc arasindaki fark cok kucuktur.

Outlier eklendikten sonra:

- Yeni MLE belirgin bicimde artmistir.
- Artis orani yaklasik %103 seviyesine ulasmistir.

Bu durum, ortalamaya dayali yontemlerin aykiri degerlerden hizli etkilendigini acikca gosteriyor.

## Yorum / Tartisma

Model, verinin yogun oldugu aralikta genel olarak iyi bir uyum veriyor. Yine de tek bir hatali kayit bile parametreyi ciddi sekilde degistirebildigi icin su onlemler gerekli:

1. Veri girisinde temel aralik kontrolleri yapilmalidir.
2. Aykiri degerler raporlanip temizleme kurali uygulanmalidir.
3. Uzun donemli veriyle model periyodik olarak guncellenmelidir.

## Proje Dosyalari

- `homework2_mle_ozgun_taslak.ipynb`: Notebook cozum dosyasi
- `ODEV2_CALISMA_PLANI.md`: Adim adim calisma plani
- `assignment_extracted.txt`: PDF'den metin cikarimi

## Kullanilan Kutuphaneler

- numpy
- scipy
- matplotlib
