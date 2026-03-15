# YZM212 Makine Öğrenmesi Ödevi - Kontrol Rehberi (README)

## Diger Odev Baglantisi
- Bu repodaki diger odevi dogrudan acmak icin: [Odev2-MLE-Trafik-Analizi](../Odev2-MLE-Trafik-Analizi)

## 1. BÖLÜM: Teorik Temeller (Viterbi Hesaplaması)
**Soru:** [High, Low] dizisi için Viterbi adımlarını hesaplayınız.
- **Döküman Kontrolü:** `report/cozum_anahtari.md` dosyasında PDF parametreleriyle (P(e)=1.0, 0.7/0.3 emisyon) adım adım el hesaplaması yapılmıştır.
- **Doğrulama Komutu:** Aşağıdaki scripti çalıştırarak rapordaki adımları programatik olarak teyit edebilirsiniz:
  ```bash
  python src/viterbi_calculation.py
  ```

---

## 2. BÖLÜM: Uygulama (Speech Classifier)
**Görev:** "EV" ve "OKUL" modellerini kurma, eğitim verisi oluşturma ve sınıflandırma fonksiyonu yazma.
- **Kod Kontrolü:** `src/hmm_model.py` dosyasında modeller, eğitim verisi yapıları ve `score` tabanlı sınıflandırma fonksiyonu yer almaktadır.
- **Doğrulama Komutu:** Sistemin bilinmeyen dizileri Log-Likelihood skoruna göre nasıl sınıflandırdığını görmek için:
  ```bash
  python src/speech_classifier_demo.py
  ```

---

## 3. BÖLÜM: Analiz ve Yorumlama
**Sorular:** Gürültü (noise) etkisi ve Deep Learning karşılaştırması.
- **Cevap Kontrolü:** `report/cozum_anahtari.md` dosyasının en sonundaki **"3. Bölüm: Analiz ve Yorumlama"** başlığı altına eklenmiştir.

---

## Dosya Yapısı ve Gönderim Formatı
- `data/`: Örnek veri dizini.
- `src/`: Yazılan Python kodları.
- `report/`: Manuel hesaplamalar ve analizlerin PDF/Markdown çıktısı.
- `requirements.txt`: Gerekli kütüphaneler (hmmlearn, numpy).
