import numpy as np
from hmmlearn import hmm

# -------------------------------------------------------------
# Makine Öğrenmesi Dersi 1. Ödev: Konuşma Tanıma
# Faz 3: "EV" ve "OKUL" Modellerinin Tasarımı ve Sınıflandırma
# -------------------------------------------------------------

# Gözlemleri temsil eden indeksler:
# 0 -> High
# 1 -> Low
OBS_HIGH = 0
OBS_LOW = 1

def create_model_ev():
    """
    Teorik kısımdaki (Faz 2) parametreleri kullanan EV modelini oluşturur.
    Durumlar (States): e, v (2 durum)
    Gözlemler: High, Low (2 gözlem)
    """
    # MultinomialHMM, ayrık gözlemler için kullanılır. 
    # n_components: Durum sayısı (2), n_iter: Eğitim iterasyonları
    model_ev = hmm.CategoricalHMM(n_components=2, random_state=42)
    
    # Başlangıç olasılıkları (pi) [P(e), P(v)]
    # PDF'de P(e)=1.0 olarak verilmiş
    model_ev.startprob_ = np.array([1.0, 0.0])
    
    # Geçiş Matrisi (A)
    # Satırlar: e, v | Sütunlar: e, v
    # PDF: P(e->e)=0.6, P(e->v)=0.4, P(v->v)=0.8, P(v->e)=0.2
    model_ev.transmat_ = np.array([
        [0.6, 0.4],  # e -> e, e -> v
        [0.2, 0.8]   # v -> e, v -> v
    ])
    
    # Emisyon Matrisi (B)
    # Satırlar: Durumlar (e, v) | Sütunlar: Gözlemler (High=0, Low=1)
    # PDF: P(High|e)=0.7, P(Low|e)=0.3, P(High|v)=0.1, P(Low|v)=0.9
    model_ev.emissionprob_ = np.array([
        [0.7, 0.3],  # e durumunda -> P(High), P(Low)
        [0.1, 0.9]   # v durumunda -> P(High), P(Low)
    ])
    
    return model_ev


def create_model_okul():
    """
    "OKUL" kelimesini temsil edecek farklı bir HMM modeli oluşturur.
    Örnek olarak 3 durumlu ('o', 'ku', 'l') ve farklı olasılıklara sahip 
    bir yapı kurgulanmıştır. (Gözlemler yine High=0, Low=1 olarak kabul ediliyor).
    """
    model_okul = hmm.CategoricalHMM(n_components=3, random_state=42)
    
    # 3 durum için Başlangıç Olasılıkları
    # Kelime genellikle 'o' ile başlayacağı için ilk durumun olasılığı yüksek seçildi
    model_okul.startprob_ = np.array([0.8, 0.1, 0.1])
    
    # 3x3 Geçiş Matrisi (A)
    # Durumların ağırlıklı olarak ileri gitme eğiliminde (Left-to-Right and self loops) olduğu varsayımı
    model_okul.transmat_ = np.array([
        [0.5, 0.4, 0.1],
        [0.1, 0.6, 0.3],
        [0.1, 0.2, 0.7]
    ])
    
    # 3x2 Emisyon Matrisi (B)
    model_okul.emissionprob_ = np.array([
        [0.3, 0.7],  # Durum 1 (o) eğilimleri
        [0.9, 0.1],  # Durum 2 (ku) eğilimleri
        [0.4, 0.6]   # Durum 3 (l) eğilimleri
    ])
    
    return model_okul


def classify_sequence(sequence, model_ev, model_okul):
    """
    Verilen gözlem dizisinin hangi HMM modeline ait olma olasılığının daha
    yüksek olduğunu bulan fonksiyon. hmmlearn'ün 'score' (log-olabilirlik) 
    fonksiyonunu kullanır.
    """
    # hmmlearn girdiyi 2D array (n_samples, n_features) olarak bekler
    seq_reshaped = np.array(sequence).reshape(-1, 1)
    
    # Score hesaplaması (Log likelihood - P(O|M))
    score_ev = model_ev.score(seq_reshaped)
    score_okul = model_okul.score(seq_reshaped)
    
    print(f"Gözlem Dizisi: {sequence}")
    print(f"EV Modeli Log-olabilirlik (Score): {score_ev:.4f}")
    print(f"OKUL Modeli Log-olabilirlik (Score): {score_okul:.4f}")
    
    if score_ev > score_okul:
        print("-> Tahmin: EV\n")
        return "EV"
    else:
        print("-> Tahmin: OKUL\n")
        return "OKUL"

if __name__ == "__main__":
    # Modellerin başlatılması
    model_ev = create_model_ev()
    model_okul = create_model_okul()
    
    # -------------------------------------------------------------
    # Ödev Madde 3: Temsili "Eğitim Verisi" (Gözlem Dizileri) Oluşturma
    # -------------------------------------------------------------
    # Not: Parametreleri manuel set ettiğimiz için fit() kullanmaya gerek yok 
    # ancak yapısal olarak veriyi tanımlıyoruz.
    train_data_ev = [
        [OBS_HIGH, OBS_LOW],
        [OBS_HIGH, OBS_HIGH, OBS_LOW],
        [OBS_HIGH, OBS_LOW, OBS_LOW]
    ]
    
    train_data_okul = [
        [OBS_LOW, OBS_HIGH, OBS_LOW],
        [OBS_LOW, OBS_LOW, OBS_HIGH],
        [OBS_LOW, OBS_HIGH, OBS_HIGH]
    ]
    
    print("----- HMM Modelleri ve Eğitim Verileri Hazır -----\n")
    
    # Test 1: Faz 2'deki teorik hesaplama örneği [High, Low] -> [0, 1]
    # EV teorik olarak: 0.1296 olasılığa sahipti. (Log10'u ~ -0.887, Ln'i ~ -2.043)
    test_seq_1 = [OBS_HIGH, OBS_LOW]
    print("Test 1: Faz 2'deki [High, Low] Örneği")
    classify_sequence(test_seq_1, model_ev, model_okul)
    
    # Test 2: 'EV' modelinin Emisyon matrisinde High(0) ve Low(1) özellikleri 
    # ardışık yoğunlukta olan bir gözlem [e(High=0.8), v(Low=0.9)] 
    test_seq_2 = [OBS_HIGH, OBS_LOW, OBS_LOW]
    print("Test 2: [High, Low, Low] (Muhtemelen EV çıkmalı)")
    classify_sequence(test_seq_2, model_ev, model_okul)
    
    # Test 3: 'OKUL' modeli Durum 2'de High(0.9) ağırlıklı olduğu için
    # High gözlemlerinin art arda çok geldiği bir dizi [0, 0, 0]
    test_seq_3 = [OBS_HIGH, OBS_HIGH, OBS_HIGH]
    print("Test 3: [High, High, High] (Muhtemelen OKUL çıkmalı)")
    classify_sequence(test_seq_3, model_ev, model_okul)
