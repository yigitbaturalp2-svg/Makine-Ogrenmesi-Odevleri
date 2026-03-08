import numpy as np
from hmm_model import create_model_ev, create_model_okul, classify_sequence

# -------------------------------------------------------------
# 2. Bölüm: Kelime Sınıflandırma (Speech Classifier) Simülasyonu
# Bu script, ödevin uygulama kısmındaki tüm gereksinimleri 
# tek bir akışta doğrular.
# -------------------------------------------------------------

def run_section_2_demo():
    print("--- 2. BÖLÜM: KELİME SINIFLANDIRICI (SPEECH CLASSIFIER) ---")
    
    # Madde 1: hmmlearn zaten yüklü varsayılıyor (requirements.txt)
    
    # Madde 2: İki farklı modelin tanımlanması
    print("\n[+] Modeller Oluşturuluyor...")
    model_ev = create_model_ev()
    model_okul = create_model_okul()
    print("    - 'EV' Modeli (2 Durumlu) [Hazır]")
    print("    - 'OKUL' Modeli (3 Durumlu) [Hazır]")

    # Madde 3: Temsili Eğitim Verisi
    print("\n[+] Temsili Eğitim Verileri (Gözlem Dizileri):")
    train_ev = [[0, 1], [0, 1, 1], [0, 0, 1]] # High-Low ağırlıklı
    train_okul = [[1, 0, 1], [1, 1, 0], [1, 0, 0]] # Farklı karakteristikte
    print(f"    - EV için örnekler: {train_ev}")
    print(f"    - OKUL için örnekler: {train_okul}")

    # Madde 4: Log-Likelihood Hesaplayan ve Sınıflandıran Fonksiyon
    # (Bu fonksiyon src/hmm_model.py içindeki classify_sequence fonksiyonudur)
    
    print("\n[+] Bilinmeyen Ses Verisi Sınıflandırma Testleri:")
    print("-" * 50)
    
    # Test Senaryosu A: "EV" karakteristiğine yakın bir dizi
    test_1 = [0, 1, 1] # High, Low, Low
    print("Senaryo 1: Dışarıdan gelen veri -> [High, Low, Low]")
    classify_sequence(test_1, model_ev, model_okul)

    # Test Senaryosu B: "OKUL" karakteristiğine yakın bir dizi
    test_2 = [0, 0, 0] # High, High, High (OKUL modelimizde High yoğunluğu var)
    print("Senaryo 2: Dışarıdan gelen veri -> [High, High, High]")
    classify_sequence(test_2, model_ev, model_okul)

    print("-" * 50)
    print("\nAnaliz: Log-Likelihood değeri (score) hangi modelde daha yüksek (0'a daha yakın)")
    print("çıkarsa, sistem o modeli 'tahmin' olarak seçmektedir.")

if __name__ == "__main__":
    run_section_2_demo()
