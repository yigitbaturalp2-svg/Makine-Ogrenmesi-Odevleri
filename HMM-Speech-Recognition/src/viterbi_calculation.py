import numpy as np
from hmmlearn import hmm

# -------------------------------------------------------------
# 1. Bölüm: Teorik Hesaplama (Viterbi) Doğrulama Scripti
# Bu script, ödev dökümanındaki parametrelerle Viterbi adımlarını 
# hem manuel simülasyonla hem de hmmlearn ile hesaplar.
# -------------------------------------------------------------

def manual_viterbi_steps():
    # Parametreler (PDF'den)
    pi_e = 1.0
    pi_v = 0.0
    
    a_ee, a_ev = 0.6, 0.4
    a_ve, a_vv = 0.2, 0.8
    
    b_e_high, b_e_low = 0.7, 0.3
    b_v_high, b_v_low = 0.1, 0.9
    
    obs = ["High", "Low"]
    
    print("--- 1. BÖLÜM: VITERBİ ADIM ADIM HESAPLAMA ---")
    
    # Adım 1: t=1 (O1=High)
    v1_e = pi_e * b_e_high
    v1_v = pi_v * b_v_high
    print(f"\nAdım 1 (t=1, Gözlem: {obs[0]}):")
    print(f"v1(e) = {pi_e} * {b_e_high} = {v1_e:.2f}")
    print(f"v1(v) = {pi_v} * {b_v_high} = {v1_v:.2f}")
    
    # Adım 2: t=2 (O2=Low)
    # v2(e)
    v2_e_from_e = v1_e * a_ee
    v2_e_from_v = v1_v * a_ve
    v2_e = max(v2_e_from_e, v2_e_from_v) * b_e_low
    
    # v2(v)
    v2_v_from_e = v1_e * a_ev
    v2_v_from_v = v1_v * a_vv
    v2_v = max(v2_v_from_e, v2_v_from_v) * b_v_low
    
    print(f"\nAdım 2 (t=2, Gözlem: {obs[1]}):")
    print(f"v2(e) = max({v1_e:.1f}*{a_ee}, {v1_v:.1f}*{a_ve}) * {b_e_low} = {max(v2_e_from_e, v2_e_from_v):.2f} * {b_e_low} = {v2_e:.4f}")
    print(f"v2(v) = max({v1_e:.1f}*{a_ev}, {v1_v:.1f}*{a_vv}) * {b_v_low} = {max(v2_v_from_e, v2_v_from_v):.2f} * {b_v_low} = {v2_v:.4f}")
    
    # Sonuç
    if v2_v > v2_e:
        result = "e -> v"
        prob = v2_v
    else:
        result = "e -> e"
        prob = v2_e
        
    print(f"\nSONUÇ:")
    print(f"En yüksek olasılık: {prob:.4f}")
    print(f"En olası fonem dizisi: {result}")
    print("-" * 45)

def logic_verification_with_hmmlearn():
    # hmmlearn ile doğrulama
    model = hmm.CategoricalHMM(n_components=2)
    model.startprob_ = np.array([1.0, 0.0])
    model.transmat_ = np.array([[0.6, 0.4], [0.2, 0.8]])
    model.emissionprob_ = np.array([[0.7, 0.3], [0.1, 0.9]])
    
    # [High(0), Low(1)]
    obs_seq = np.array([[0], [1]])
    logprob, path = model.decode(obs_seq, algorithm="viterbi")
    
    # path: 0 -> e, 1 -> v
    path_str = " -> ".join(["e" if s == 0 else "v" for s in path])
    print(f"\nhmmlearn 'decode' Sonucu (Viterbi Path): {path_str}")
    print(f"Log-Olasılık: {logprob:.4f}")

if __name__ == "__main__":
    manual_viterbi_steps()
    logic_verification_with_hmmlearn()
