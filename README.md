# CLOAF yeniden implementasyonu: yol haritası

Proje: *Correcting Self Intersections in SMPL* (supervisor: Tony Wang, TUM)
Makale: Davydov ve ark., *CLOAF: CoLlisiOn-Aware Human Flow*, CVPR 2024 (arXiv 2403.09050)

---

## 0. Önce iki küçük ama önemli adım

1. **Yazarlara e-posta at.** Andrey Davydov'a (`andrey.davydov@epfl.ch`, makaledeki `{name}.{surname}@epfl.ch` kalıbına göre) yaz: kod ya da eğitilmiş `f_ω` checkpoint'i paylaşılabilir mi, ve aşağıdaki belirsizlikler (Bölüm 4). Cevap gelmezse bile, sorduğunu supervisor'a göstermen işine yarar.
2. **Supervisor'la "eşleşme" kriterini netleştir.** "Değerler uyuşmalı" ne demek: ±1 mm MPJPE mi, yoksa trend aynı olsun mu? Önerim: aşağıdaki **Doğrulama Tablosu**nu ona göster ve hangi satırların zorunlu olduğunu birlikte seçin.

---

## 1. Platform: nerede çalışmalısın?

Önerim **hibrit bir kurulum**: kod git deposunda `.py` modülleri olarak dursun, ağır işler GPU sunucusunda koşsun, Jupyter sadece görselleştirme için kullanılsın.

| Ne | Nerede | Neden |
|---|---|---|
| Kod yazma | Lokalde VS Code + git (bu klasör) | Eğitim script'leri notebook'ta yönetilmez. Modül yapısı test etmeyi ve sunucuya taşımayı kolaylaştırır |
| Eğitim (AMASS) ve 3DPW değerlendirmesi | **TUM GPU sunucusu** (chair'in sunucusu ya da LRZ AI Systems; Tony Wang'a sor) | Makale A100 kullanmış. Jacobian, SMPL'i çift hassasiyette yüzlerce kez çağırıyor |
| Küçük deneyler (Adım 1-2) | Lokal CPU yeterli, GPU varsa daha iyi | Eğitim gerektirmiyor |
| Animasyon izleme | Lokalde, ekranlı makinede **aitviewer** ya da `notebooks/` içinde Jupyter | Proje tanımı "sonuçları animasyon olarak incele" diyor |
| Google Colab | Sadece yedek seçenek | Oturumlar kopuyor. Ayrıca SMPL/AMASS lisans dosyalarını Drive'a yüklemen gerekiyor |

Ortam kurulumu:

```bash
conda create -n cloaf python=3.11 -y
```

```bash
conda activate cloaf
```

```bash
pip install -r requirements.txt
```

Kayıt olup indirmen gereken veriler (hepsi akademik lisanslı):

- **SMPL** (smpl.is.tue.mpg.de): `data/body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl`
- **AMASS** (amass.is.tue.mpg.de): eğitim verisi, SMPL+H formatında
- **3DPW** (virtualhumans.mpi-inf.mpg.de/3DPW): değerlendirme
- **4D-Humans / HMR2.0** (github.com/shubham-goel/4D-Humans): baseline tahminleri
- **J_regressor_h36m.npy**: SPIN'in data paketinden; standart 14 eklemli 3DPW metriği için

---

## 2. Kod yapısı

```
cloaf/
  body.py         SMPL sarmalayıcı, yüzey örnekleme (S=1000, kasık ve koltuk altı hariç),
                  X(Θ), Jacobian (sonlu fark veya jacfwd, float64), J^† f
  collision.py    Generalized Winding Number ile öz-kesişim tespiti, Col.Rate@C
  model.py        f_ω: Fourier features (n_f=20) + 6 katmanlı concat-MLP (ShapeFlow tarzı),
                  girdi (x, Θ(t), Θ1, Δt), çıktı hızla ölçekli
  train.py        AMASS hazırlığı, çift örnekleme, yörünge kaybı (Denk. 9, Θ~ lineer interp.)
  integrate.py    ODE (Denk. 6) + dopri5, basit hedef alanı (Denk. 10-11, C.14),
                  dizi düzeltme: mode="all" (makale) / "colliding_only" (uzantı), Jitter
  metrics.py      MPJPE, PA-MPJPE, Accel.Err, temporal jerk
scripts/
  check_linearity.py   ADIM 1: Fig. 5 ve B.7/B.8 (eğitim gerekmez)
  demo_target_field.py ADIM 2: Fig. 3/4 (eğitim gerekmez)
  eval_3dpw.py         ADIM 3/5: Table 1, 2, 4
tests/smoke_test.py    Sahte bir modelle uçtan uca test (SMPL dosyası gerekmez)
```

Kodu SMPL dosyası olmadan şu komutla test edebilirsin:

```bash
python tests/smoke_test.py
```

Bu testi sahte bir elipsoid modelde çalıştırıp geçirdim. Fd ile jacfwd Jacobian'ı arasındaki fark ~1e-10 çıktı, GWN değerleri de beklendiği gibi: merkez ≈ 1, yüzey ≈ 0.5.

---

## 3. Yol haritası

Adımlar, **bir sonraki adım bir öncekinin sayılarına güvenecek** şekilde sıralandı.

### Adım 1 (1. hafta): SMPL'in lineerliği ve Jacobian (eğitim yok)

```bash
python scripts/check_linearity.py --smpl data/body_models/smpl
```

Hedef: Fig. 5 ve B.7/B.8 eğrileri. Bunlar ilk eşleşen sayıların olur. Tutmazsa sorun Jacobian'da ya da Θ tanımındadır.

### Adım 2 (1-2. hafta): Basit alanla entegrasyon (eğitim yok)

```bash
python scripts/demo_target_field.py --smpl data/body_models/smpl
```

```bash
python scripts/demo_target_field.py --smpl data/body_models/smpl --box
```

Hedef: Fig. 3 ve Fig. 4. El hedefe gitmeli. Kutu varsa kutunun yüzeyinde durmalı. Sonuçları aitviewer'da izle.

### Adım 3 (2-3. hafta): Öz-kesişim metriğini kalibre et

HMR2.0'ı 3DPW-test üzerinde çalıştır, sonra:

```bash
python scripts/eval_3dpw.py --smpl data/body_models/smpl --pw3d data/3DPW --pred data/hmr2_3dpw_test.npz --no_cloaf
```

Hedef: Table 1/2'deki HMR2.0 satırı. **Bu adım kritik.** Makale tespit yöntemini açıklamıyor. Col.Rate değerin 39.2%'yi tutmuyorsa GWN eşiğini, hariç bölgeleri ve kullanılan frame'leri ayarla. Bunu CLOAF'tan önce çöz. Yoksa CLOAF'un sonuçlarını yorumlayamazsın.

### Adım 4 (3-6. hafta): `f_ω` eğitimi

```bash
python -m cloaf.train --smpl data/body_models/smpl --amass data/amass_smpl_30fps.npz
```

`prepare_amass()` fonksiyonu ile AMASS'ı tek bir npz dosyasına çevir. Önce AMASS'ın küçük bir alt kümesinde (ör. CMU) aşırı öğrenmeyi (overfit) dene.

### Adım 5 (6-7. hafta): CLOAF ile 3DPW

`eval_3dpw.py` script'ini `--field checkpoints/field.pt --mode all` ile çalıştır. Hedef: Table 2'deki CLOAF satırı ve Table 4.

### Adım 6 (7. haftadan sonra): Uzantı (proje tanımından)

`--mode colliding_only` ile `--mode all`'ı karşılaştır:

- Col.Rate
- MPJPE / Accel
- düzeltilen frame ile düzeltilmeyen frame arasındaki **geçişlerde** `temporal_jerk`
- aitviewer'da yan yana video

---

## 4. Doğrulama Tablosu (supervisor'a göster)

| Deney | Makale değeri | Kaynak | Bizim değer |
|---|---|---|---|
| RE, ‖δΘ‖=1e-2, S=6890 | ≈ 7.1e-2 | Fig. B.8 | |
| RE, ‖δΘ‖=1e-1, S=6890 | ≈ 7.0e-1 | Fig. B.8 | |
| RE, S=10 / S≥100 / S=1000 | ≈0.7 / ≈0.05–0.08 / <0.1 | Fig. 5 | |
| Jacobian süresi, S=1000 / 6890 | ~15–20 ms / ~320 ms (A100) | Fig. 5 | |
| HMR2.0 Col@0 / @100 | 39.2% / 21.6% | Table 1 | |
| PARE, TUCH, EFT, SPIN Col@0 | 36.1 / 23.7 / 17.8 / 15.5 % | Table 1 | (opsiyonel) |
| HMR2.0 MPJPE / P-MPJPE / Accel | 82.0 / 52.7 / 16.1 | Table 2 | |
| **CLOAF** MPJPE / P-MPJPE / Accel / Col@0 | **82.3 / 53.2 / 9.4 / 0.0%** | Table 2 | |
| CLOAF-Jitter | 81.6 / 52.6 / 15.9 | Table 4 | |
| CLOAF-Keyposes (K=128) | 82.9 / 53.6 / 16.4 | Table 4 | |
| EFT + CLOAF fine-tune (diff) | P-MPJPE 55.4, Col 3.4% | Table 3 | (kapsam dışı olabilir) |

### Makalede belirsiz ya da tutarsız olan noktalar

Bunları varsayım olarak raporla ve supervisor'a sor:

1. **HMR2.0 P-MPJPE değeri iki tabloda farklı:** Table 1'de 54.3, Table 2'de 52.7. Table 1'deki değer muhtemelen HMR2.0 makalesinden alınmış, Table 2'deki ise yazarların kendi çalıştırması.
2. **Öz-kesişim tespiti nasıl yapılıyor?** Makale sadece "vücudun içindeki vertex sayısı > C" diyor. Biz GWN > 1.0 kullanıyoruz.
3. **Table 1'deki "@100 (w/o col.)" sütunu** belirsiz tanımlanmış.
4. **Ağ mimarisi:** gizli katman genişliği, aktivasyon, ağın nokta-bazlı olup olmadığı ve Fourier frekans aralığı verilmemiş. Bunlar `model.py`'deki varsayımlar.
5. **Kasık ve koltuk altı maskesi:** sadece "[26]'yı takip ederek hariç tutuldu" deniyor. `body.py` basit bir yarıçap maskesi kullanıyor. Bunu görsel olarak kontrol et.
6. **AMASS eğitim detayları:** frame aralığı ("consecutive poses, Δt_max=1"), fps, betas (SMPL+H'den SMPL'e), lr, iterasyon sayısı, batch boyutu ve M verilmemiş.
7. **Shape β ve global orient:** Θ 69 boyutlu (Fig. B.8). β ve global orient hedef frame'den alınıyor. Θ0 farklı bir β ile çakışabilir, bu yüzden `correct_sequence` bunu tekrar kontrol ediyor.
8. **Dopri5 toleransları** ve sayılarının kaç kez tekrarlandığı belirtilmemiş.
9. **İlk frame:** çakışmalıysa ne yapıldığı belirsiz. Biz önce Jitter uyguluyoruz, sonra CLOAF çalıştırıyoruz.

---

## 5. Taslak koddaki bilinen eksikler (TODO)

- `eval_3dpw.py`: Accel hesabında `campose_valid` boşluklarını parçalara ayırmak gerekiyor. HMR2.0 çıktısını npz'ye yazan dönüştürücüyü de yazman gerekiyor.
- Keypose stratejisi (K-means, K=128) iskelet halinde. Makale keypoint mesafesi kullanıyor.
- Differentiable fine-tuning (Table 3) için `odeint_adjoint` gerekiyor. Bu muhtemelen kapsam dışı.
- Sayıların hızlanması için `jacobian(method="jacfwd")`'i gerçek SMPL'de fd ile karşılaştır ve hızlı olanı seç.
