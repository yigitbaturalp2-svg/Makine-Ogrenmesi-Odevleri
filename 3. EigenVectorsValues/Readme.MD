# Matris Manipülasyonu, Özdeğer ve Özvektörler Raporu

## Soru 1: Kavramsal Tanımlar ve Makine Öğrenmesi ile İlişkisi

**Matris Manipülasyonu:** Verilerin satır ve sütunlar halinde düzenlendiği matematiksel yapılar olan matrisler üzerinde yapılan toplama, çıkarma, çarpma, devrik (transpose) alma veya ters (inverse) alma gibi işlemlere matris manipülasyonu denir. Yapay zeka ve derin öğrenmede veri setleri, ağırlıklar ve girdiler matrislerle ifade edilir, böylece büyük ölçekli işlemler CPU/GPU üzerinde vektörize edilerek çok daha hızlı paralel olarak yapılabilir.

**Özdeğer (Eigenvalue) ve Özvektör (Eigenvector):** Kare matrisler için tanımlanan bu kavramlar, doğrusal dönüşümleri anlamanın boyutudur. Bir $A$ matrisi (doğrusal bir dönüşüm fonksiyonu), sıfırdan farklı bir $v$ vektörüne uygulandığında, eğer sonuç sadece bu vektörün skaler bir katı olan $\lambda \cdot v$ çıkıyorsa (yani vektörün yönü değişmeyip sadece büyüklüğü etkileniyorsa), $v$ vektörüne **özvektör**, bu büyütme/küçültme faktörü olan $\lambda$ değerine ise **özdeğer** denir. Kısaca, $A \cdot v = \lambda \cdot v$ bağıntısıyla ifade edilir.

### Makine Öğrenmesi ile İlişkisi ve Kullanıldığı Yaklaşımlar

Makine öğrenmesinde özdeğer ve özvektörler boyutları yönetmek, sinyali gürültüden ayırmak ve veri matrislerinin analizini yapmak için temel rollere sahiptir:

1. **Temel Bileşenler Analizi (PCA - Principal Component Analysis):** Yüksek boyutlu verileri anlamlı daha alt boyutlara sıkıştırmak, overfitting'den (aşırı öğrenme) kaçınmak ve model eğitimini hızlandırmak için kullanılır. Algoritma arka planda veri setinin kovaryans matrisini oluşturur ve bu matrisin özdeğerleri ile özvektörlerini hesaplar. En yüksek özdeğerlere sahip olan özvektörler, verideki varyansın (çeşitliliğin) en fazla olduğu yönleri (temel bileşenleri) temsil eder.
2. **Tekil Değer Ayrışımı (SVD - Singular Value Decomposition):** Hem tavsiye sistemlerinde (örneğin Netflix ve Amazon'un öneri motorları) hem de doğal dil işlemede (LSA) çok sık kullanılır. Veri matrisini özdeğer-özvektör matris analizine benzer yollarla low-rank (düşük rütbeli) alt matrislere parçalayarak gizli öznitelikleri bulmayı sağlar.
3. **Spectral Clustering (Spektral Kümeleme):** Veri örneklerini graf benzeri düğüm ve bağ (benzerlik) ilişkilerine dönüştürür. Ortaya çıkan Laplacian matrisinin özdeğerleri ve özvektörleri hesaplanarak karmaşık şekilli kümeler gruplandırılır (örneğin k-means'in zorlandığı yerlerde).
4. **PageRank ve Markov Zincirleri:** Arama motorlarının web sitelerini sıraladığı mantık, bir geçiş ve bağlantı graf matrisinin temel (en büyük) özvektörünü hesaplamaktır.

**Referanslar (Kaynaklar):**
- Brownlee, J. (2018). *Introduction to Matrices and Matrix Arithmetic for Machine Learning*. Machine Learning Mastery.
- Brownlee, J. (2018). *A Gentle Introduction to Eigendecomposition, Eigenvalues, and Eigenvectors for Machine Learning*. Machine Learning Mastery. 

## Soru 2: NumPy ile Özdeğer ve Özvektör Hesaplama

NumPy'ın popüler `linalg` (linear algebra) modülü, kare matrislerin özdeğer ve özvektörlerini bulmak için hazır `eig` fonksiyonunu sunmaktadır. Ödevdeki Jupyter Notebook (`EigenVectorsValues.ipynb`) içerisinde örnek bir matris ile çalıştırılmış ve sonuçlar alınmıştır.

### Dokümantasyon ve Kaynak Kodun İncelenmesi

1. **Dokümantasyon (Belgelendirme):**  
   [NumPy 2.1 numpy.linalg.eig](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html) adresindeki döküman incelendiğinde şu detaylar öne çıkmaktadır:
   - Parametre olarak bir kare matris (veya kare matrisler dizisi) alır.
   - İki değer döndürür: `w` (özdeğerler) ve `v` (özvektörler). Dönüşteki her `w[i]` değeri, `v[:, i]` şeklindeki sütun vektörüne denktir.
   - İşlemin duyarlı (real) veya karmaşık (complex) değerleri hesaplayabileceği belirtilir.

2. **Kaynak Kod Analizi (Under the Hood):**  
   [NumPy GitHub linalg klasörü](https://github.com/numpy/numpy/tree/main/numpy/linalg) adresindeki `linalg.py` ve C uzantı modülleri (.c.src) incelendiğinde;
   - `numpy.linalg.eig` fonksiyonu tamamen saf bir Python implementasyonu (örneğin sıfırdan while döngüsüyle yazılan bir hesaplama) **değildir**.
   - Arka planda son derece optimize yapıda olan **LAPACK (Linear Algebra PACKage)** adlı endüstri standardı olan Fortran/C kütüphanesi kullanılmaktadır.
   - Çekirdek iş parçacığında `LAPACK` in `geev` rutini (genelleştirilmiş kare matris özdeğer hesabı rutini) çağrılmaktadır. Matrisin simetrik / pozitif tanımlı olma durumlarına göre de gerekirse `syevd` gibi alternatif uygun rutinlere dallanma (dispatch) yapısı barındırır.
   - Bu durum, Python ile yazılan matris kodlarının aslında C/Fortran hızında (donanımsal vektör optimizasyonlarıyla) çalıştığı makine öğrenmesi algoritmalarının performans sırlarından biridir.

## Soru 3: Hazır Fonksiyon Kullanmadan Özdeğer Hesaplama

Verilen referans kaynak ([LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors)) incelendiğinde, temel matematiksel yaklaşım olarak **Karakteristik Denklem (Characteristic Equation)** yöntemi kullanılmıştır. 

1. **Mantığı:** Geleneksel olarak özdeğerler, $det(A - \lambda \cdot I) = 0$ denkleminin çözümü (kökleri) ile bulunur.
2. **Uygulanışı (Referans Depo):** Referans koddaki yaklaşımda, öncelikle her bir otonom eleman $\lambda$'ya bağlı polinomlar şeklinde (liste formunda) ifade edilmiş ve matrisin determinantı alınırken bu alt polinomlar kuralına (FOIL vb.) uygun olarak birbiri ile çarpılmış ve toplanmıştır. Sonuçta Karakteristik Denklemin katsayı dizisi (örneğin $\lambda^2 - 7\lambda + 10 = 0$ için `[1, -7, 10]`) elde edilmiş ve `numpy.roots` fonksiyonuyla sadece kökler bulunarak `eig` komutuna gerek kalmadan özdeğer çekilmiştir.
3. **Karşılaştırma ve Sonuçlar:** Jupyter Notebook içerisine uyguladığımız kodlarda görülebileceği üzere, $2x2$ formundaki test matrisimiz için her iki yaklaşım (NumPy'ın hazır `eig` rutini ve manuel Karakteristik denklem kök bulucusu) aynı sonuçları vermiştir. Fakat hazır fonksiyon olan `numpy.linalg.eig`, C seviyesinde LAPACK rutinlerini barındırdığı için büyük çaplı ($1000x1000$ vb.) matrislerde polinom çıkarımı gibi son derece yavaş $O(N!)$ karmaşıklığa kayan determinant tabanlı manuel yaklaşıma göre kıyaslanamayacak derecede yüksek hıza ve sayısal kararlılığa sahiptir. Zaten makine öğrenmesinde bu yüzdendir ki standart paketler tercih edilir.

**Not:** Kod ve karşılaştırma çıktıları `EigenVectorsValues.ipynb` içindedir.
