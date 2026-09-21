# Catatan Hyperparameter Tuning — Kompetisi Data Mining

2026-09-21 · @Someone

## Sebelum masuk ke daftar knob

Pekerjaan kalian sudah di atas rata-rata peserta. Delapan versi submission yang terdokumentasi, tabel eksperimen gagal beserta angkanya, imputasi yang di-fit per fold supaya bebas kebocoran, dan nested validation untuk bobot kelas — itu praktik yang benar, dan banyak tim tidak sampai ke sana. Catatan ini bukan koreksi terhadap kerapian kerja kalian, melainkan soal **ke mana tenaga tuning diarahkan**.

Satu diagnosis perlu kalian tahu lebih dulu, karena mengubah hampir semua saran di bawah: **model kalian kebesaran untuk masalahnya.**

Bukti pertama, saat kapasitas model dinaikkan bertahap, skor cross-validation justru turun terus:

| Kapasitas | Konfigurasi | Macro-F1 CV |
| --- | --- | --- |
| kecil | LGBM `num_leaves=7, n=300` | 0.6564 |
| sedang | LGBM `num_leaves=31, n=600` | 0.6481 |
| besar | LGBM `num_leaves=127, n=1200` | 0.6426 |

Bukti kedua lebih telak: **Logistic Regression biasa** dengan regularisasi `C=0.1` mencapai 0.6684 — praktis menyamai CatBoost `depth=6` kalian yang sudah dituning Optuna. Menambahkan spline untuk melonggarkan asumsi linearitas malah menurunkan skor ke 0.6583.

Artinya hubungan fitur-ke-target di dataset ini hampir sepenuhnya **aditif dan monoton**. Tidak ada interaksi rumit yang perlu ditangkap. Ketika `CAT_PARAMS` mendarat di `depth=6` dengan `iterations=718` untuk 4048 baris, sebagian besar kapasitas itu dipakai menghafal noise, bukan mempelajari pola.

Konsekuensinya: menurunkan `depth` dari 6 ke 3 memberi **+0.0077** macro-F1 (nested CV), dan diverifikasi ulang di pembagian fold yang sama sekali berbeda menghasilkan **+0.0056**. Itu lebih besar daripada gabungan seluruh feature engineering yang sudah kalian kerjakan.

## Langkah nol: ukur dulu, baru tuning

Kebiasaan yang perlu dibangun: sebelum menyapu hyperparameter apa pun, **cari tahu dulu apakah masalahnya kekurangan kapasitas atau kelebihan kapasitas**. Kalau salah menebak, seluruh pencarian akan diarahkan ke wilayah yang keliru — persis yang terjadi pada v1–v8.

Caranya murah: latih tiga sampai empat model dengan kapasitas yang sengaja dibedakan jauh, lalu lihat arah kurvanya.

```python
for tag, kw in [("kecil",  dict(num_leaves=7,   n_estimators=300,  learning_rate=0.05)),
                ("sedang", dict(num_leaves=31,  n_estimators=600,  learning_rate=0.03)),
                ("besar",  dict(num_leaves=127, n_estimators=1200, learning_rate=0.02))]:
    skor = evaluasi_cv(LGBMClassifier(**kw, class_weight='balanced'))
    print(tag, round(skor, 4))
```

Cara membacanya:

1. **Kurva naik** (besar lebih baik) — model masih kurang kapasitas. Sapu ke arah lebih dalam, lebih banyak iterasi, regularisasi lebih longgar.
2. **Kurva turun** (kecil lebih baik) — model kelebihan kapasitas. Sapu ke arah lebih dangkal dan regularisasi lebih kuat. **Ini kasus kalian.**
3. **Kurva datar** — hyperparameter bukan lever utama; cari gain di preprocessing atau fitur.

Uji pembanding kedua yang juga murah: jalankan **Logistic Regression** sebagai baseline. Kalau model linear sederhana bisa menempel di angka gradient boosting kalian, itu tanda kuat bahwa polanya aditif dan model besar cuma menambah varians. Di laporan kalian, LogReg tercatat 0.645 dan CatBoost 0.665 — selisih 0.02 itu sinyal yang terlewat dibaca.

Biasakan menjalankan dua diagnostik ini di awal setiap proyek tabular. Lima belas menit di sini menghemat berjam-jam Optuna di ruang pencarian yang salah.

## Lapisan 1 — parameter model CatBoost

Di blok `%% 2. Konfigurasi` notebook kalian:

```python
CAT_PARAMS = {'border_count': 167, 'depth': 6, 'iterations': 718,
              'l2_leaf_reg': 7.94, 'learning_rate': 0.0241}
```

Lima parameter ini yang sudah disapu RandomizedSearchCV lalu Optuna. Masalahnya bukan kalian tidak menyapu — **ruang pencariannya yang salah**. Pencarian mendarat di `depth=6`, dan kemungkinan besar range `depth` kalian dibatasi 4–10, sehingga wilayah yang benar tidak pernah dicoba sama sekali.

Range yang saya sarankan untuk sapuan ulang:

| Parameter | Range lama (dugaan) | Range yang disarankan | Catatan |
| --- | --- | --- | --- |
| `depth` | 4–10 | **2–6** | knob terpenting; optimum ada di 3 |
| `iterations` | 300–1000 | 200–900 | makin dangkal, makin butuh banyak iterasi |
| `learning_rate` | 0.01–0.1 | 0.02–0.10 | pasangan alami `iterations` |
| `l2_leaf_reg` | 1–10 | 1–30 | rem overfit utama setelah depth |
| `border_count` | 32–255 | 32–255 | dampak kecil, boleh dibiarkan |

Ada satu kesalahan halus yang perlu kalian tahu, dan ini tertulis di komentar kode kalian sendiri di v7: hyperparameter itu dituning saat fitur masih **one-hot**, lalu dipakai ulang setelah pindah ke **native categorical**. Ruang optimumnya berubah saat representasi fiturnya berubah. Kalau ganti encoding, tuning harus diulang — bukan diwariskan.

### Parameter yang belum pernah disentuh

Di `make_model()` hanya lima parameter di atas yang diteruskan. Yang tersedia tapi dibiarkan default:

- `random_strength` — menambah keacakan pada pemilihan split; rem overfit yang relevan untuk data kecil
- `min_data_in_leaf` — minimum sampel per daun; langsung membatasi hafalan
- `rsm` — proporsi fitur yang dipakai tiap level (setara `colsample_bytree`)
- `bagging_temperature` — kekuatan bootstrap Bayesian
- `one_hot_max_size` — batas kategori yang di-one-hot otomatis alih-alih pakai target statistics
- `grow_policy` — `SymmetricTree` (default), `Depthwise`, atau `Lossguide`
- `loss_function` — `MultiClass` vs `MultiClassOneVsAll`

Untuk data yang polanya sesederhana ini, `random_strength` dan `min_data_in_leaf` adalah dua yang paling layak dimasukkan ke ruang pencarian setelah `depth`.

## Lapisan 2 — imbalance dan aturan keputusan

### `AUTO_CLASS_WEIGHTS` tidak pernah masuk pencarian

```python
AUTO_CLASS_WEIGHTS = 'Balanced'
```

Ini variabel tersendiri di config kalian, tapi nilainya tidak pernah disapu. Pilihannya `'Balanced'`, `'SqrtBalanced'`, atau `None`.

Yang penting dipahami: kalian sekarang mengoreksi ketidakseimbangan kelas **dua kali** — sekali lewat `auto_class_weights` saat training, sekali lagi lewat bobot pasca-training sebelum `argmax`. Koreksi saat training mendistorsi probabilitas keluaran model, padahal tuning ambang justru butuh probabilitas yang terkalibrasi baik.

Kombinasi `None` + bobot pasca-training saja belum pernah kalian uji, dan secara teori itu yang lebih tepat untuk optimasi macro-F1. Perlakukan `AUTO_CLASS_WEIGHTS` sebagai hyperparameter kategorikal tiga nilai di dalam ruang pencarian.

### Aturan keputusan punya knob-nya sendiri

```python
W_GRID   = np.round(np.arange(0.5, 3.01, 0.05), 2)
MIN_GAIN = 0.002
def tune_weights(P, y, top_frac=0.01):
```

Tiga hal yang bisa disapu di sini:

- **rentang dan langkah `W_GRID`** — 0.5–3.0 dengan langkah 0.05 menghasilkan 2.500 kombinasi; grid sehalus itu di atas objektif yang ber-sd 0.0045 lebih banyak menangkap noise daripada sinyal
- **`top_frac`** — berapa persen kombinasi terbaik dirata-ratakan. Nilai kalian 0.01 (25 kombinasi teratas). Menaikkannya ke 0.03–0.05 membuat bobotnya lebih tahan noise, karena tidak menempel pada satu puncak kebetulan
- **`MIN_GAIN`** — ambang kapan bobot dipakai; 0.002 itu pilihan yang belum pernah diuji sensitivitasnya

Lapisan ini layak mendapat perhatian lebih besar daripada yang kalian berikan. Laporan kalian sendiri mencatat reweighting pasca-training sebagai satu-satunya teknik yang terbukti reliable menaikkan skor — tapi knob-nya justru dibiarkan pada nilai bawaan, sementara Optuna dihabiskan di `CAT_PARAMS`.

Prinsip umum yang layak dibawa ke proyek lain: **pisahkan estimasi probabilitas dari aturan keputusan.** Tugas model adalah menghasilkan probabilitas sebaik mungkin; menentukan ambang mana yang memaksimalkan metrik adalah pekerjaan terpisah yang dituning di OOF. Mencampur keduanya — misalnya memakai class weight saat training *dan* bobot saat prediksi — membuat keduanya saling mengganggu.

## Lapisan 3 — keputusan yang tidak terlihat seperti hyperparameter

Ini bagian yang paling sering dilewatkan orang. Banyak baris di `clean_rows()`, `Preprocessor`, dan `add_features()` terlihat seperti logika tetap, padahal isinya **pilihan** — dan setiap pilihan bisa dimasukkan ke ruang pencarian.

### Keputusan pembersihan

```python
df.loc[m, 'screen_time_hrs_per_day'] = df.loc[m, 'screen_time_hrs_per_day'] / 10
m = (df['sleep_hours'] < 0) | (df['sleep_hours'] > 24)
```

Pembagi `/10` itu asumsi, bukan fakta. Perlakukan sebagai hyperparameter kategorikal: `/10`, `/3`, atau set NaN.

Dan sedikit catatan soal asumsi itu sendiri, karena ini pelajaran investigasi data yang bagus. Nilai yang rusak adalah 26.7 sampai 48.0, sementara kolom bersihnya berentang 1.0–21.6 dengan median 8.9 dan **seluruhnya berpresisi satu angka desimal**. Dibagi 10 hasilnya 2.67–4.80 — dua angka desimal, dan terlempar ke kuartil terbawah. Dibagi 3 hasilnya 8.9–16.0 — satu angka desimal, tepat di tengah-atas distribusi, dan nilai terkecilnya persis sama dengan median kolom bersih. Korupsinya kemungkinan besar perkalian ×3, bukan pergeseran desimal.

Ada juga anomali yang terlewat sepenuhnya: **`notifications_per_day` punya 35 nilai negatif** (minimum −5). Itu kelas kesalahan yang sama dengan `"52 yrs"` dan `screen_time > 24`, jelas ditanam sengaja oleh panitia, tapi masuk ke model apa adanya.

Pelajarannya: jangan hanya membersihkan anomali yang disebutkan di deskripsi soal. Jalankan `.describe()` pada setiap kolom numerik dan tanyakan apakah nilai minimum dan maksimumnya masuk akal secara fisik.

### Strategi imputasi

```python
MEAN_COLS   = ['caffeine_intake_mg_per_day', 'class_attendance_percent', 'social_media_hrs_per_day']
MEDIAN_COLS = ['study_hours_per_week', 'age']
MODE_COLS   = ['uses_productivity_app']
ADD_MISSING_FLAGS = False
```

Penugasan kolom ke mean/median/mode adalah keputusan manual yang bisa disapu. Dan ada opsi keempat yang tidak ada di daftar kalian: **tidak diimputasi sama sekali**, biarkan CatBoost menangani NaN secara native. CatBoost memilih arah split terbaik untuk nilai hilang — lebih informatif daripada menimpanya dengan mean, yang justru menghapus beda antara "nilainya rendah" dan "nilainya tidak diketahui".

`ADD_MISSING_FLAGS` sudah kalian sediakan sebagai boolean tapi hanya diuji sekali. Untuk dataset ini kesimpulannya memang benar — missing-rate-nya rata di ketiga kelas (sekitar 4–5% di semua), jadi mekanismenya MCAR dan flag-nya tidak membawa informasi. Tapi cara kalian menyimpulkannya kebetulan benar, bukan karena diuji. Lain kali, cek dulu missing-rate per kelas sebelum memutuskan.

### Konstanta di `add_features()`

```python
d['sleep_stress_ratio'] = d['sleep_hours'] / (d['academic_stress_level_1_10'] + 1)
d['study_eff']          = d['gpa'] / (d['study_hours_per_week'] + 1)
```

Angka `+1` itu smoothing yang dipilih arbitrer. Selain itu, on/off tiap fitur turunan (lima boolean) juga ruang pencarian tersendiri — ini praktis menjadikan feature selection sebagai bagian dari HPO.

Satu peringatan berdasarkan pengukuran: di dataset ini, membuang kolom yang korelasinya rendah justru **menurunkan** skor (0.6614 vs 0.6681). CatBoost menanganinya baik-baik saja dan ternyata masih ada sedikit sinyal di sana. Jangan buang fitur hanya karena korelasi Spearman-nya kecil — uji dulu.

## Yang BUKAN hyperparameter

```python
N_SPLITS = 5
N_REPEATS = 2
SEED = 42
```

Tiga variabel ini **alat ukur**, bukan knob. Ini pembedaan konseptual yang penting dan mudah dilanggar tanpa sadar.

Menaikkan `N_REPEATS` boleh dan bagus — estimasi jadi lebih stabil, sd antar-repeat mengecil, keputusan kalian lebih bisa dipercaya. Tapi jangan pernah memilih nilai `N_SPLITS`, `N_REPEATS`, atau `SEED` **karena nilai itu menghasilkan skor terbaik**. Itu bukan tuning, itu memilih penggaris yang hasil bacaannya paling enak dilihat.

Ada satu kesalahan terkait yang sudah terjadi di v1–v8: **harness evaluasinya berganti-ganti antar versi.** v1–v6 pakai single 5-fold, v7–v8 pakai repeated plus nested. Membandingkan 0.6643 (v2) dengan 0.6692 (v3) dengan 0.6647 (v7) lintas protokol berbeda sama saja dengan membandingkan noise.

Aturan praktisnya: **tetapkan satu fungsi evaluasi di awal proyek, pakai untuk semua eksperimen, jangan diubah lagi.** Kalau harness-nya berubah di tengah jalan, semua perbandingan sebelumnya harus dijalankan ulang.

```python
HARNESS = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

def evaluasi(model, X, y):
    # ... hitung OOF, kembalikan mean DAN sd antar repeat
    return mean_skor, sd_skor
```

Selalu laporkan **mean ± sd**, bukan satu angka. Dan terima perubahan hanya kalau gain-nya melebihi sekitar 2× sd. Dengan sd di sekitar 0.005, artinya gain di bawah 0.01 dari satu run tidak boleh dipercaya.

## Jebakan terbesar: makin banyak trial, makin menipu

Ini bagian yang paling penting dari seluruh catatan ini, dan paling jarang diajarkan.

Skor cross-validation punya noise. Di dataset ini, sd antar-konfigurasi terukur sekitar **0.0045**. Ketika kalian menjalankan Optuna dan mengambil trial dengan skor tertinggi, sebagian dari skor itu adalah keberuntungan — dan makin banyak trial, makin besar porsi keberuntungannya.

Simulasinya, dengan asumsi semua trial punya kemampuan sejati yang sama:

| Jumlah trial | Inflasi skor CV | CV yang terlihat (skill sejati 0.6680) |
| --- | --- | --- |
| 10 | +0.0069 | 0.6749 |
| 50 | +0.0101 | 0.6781 |
| 100 | +0.0113 | 0.6793 |
| 200 | +0.0124 | 0.6804 |
| 500 | +0.0137 | 0.6817 |

Baca tabel itu pelan-pelan. Optuna 200 trial akan menampilkan CV sekitar 0.680 untuk model yang kemampuan sebenarnya 0.668. Kalian akan merasa naik 0.012, padahal tidak naik sama sekali. Lalu saat disubmit, leaderboard memberi angka yang lebih rendah — dan mudah sekali salah menyimpulkan itu sebagai "overfitting terhadap data train", padahal yang terjadi adalah overfitting terhadap **proses seleksi** kalian sendiri.

Pola yang sama juga berlaku di leaderboard. Tim kalian punya 16 entri; mengambil skor LB tertinggi dari 16 undian juga menyeleksi keberuntungan. Dengan sd leaderboard sekitar 0.016 pada 1012 baris, angka 0.67113 kalian kemungkinan besar undian terbaik, bukan model terbaik.

### Dua pengaman yang wajib

1. **Batasi jumlah trial.** Untuk ruang pencarian sekecil ini, 50 trial sudah cukup. 500 trial tidak memberi model yang lebih baik, hanya angka yang lebih menipu.
2. **Verifikasi pemenangnya di split CV yang sama sekali baru.** Setelah Optuna memilih konfigurasi terbaik, jalankan ulang konfigurasi itu dengan `random_state` berbeda dan bandingkan lagi dengan baseline. Kalau gain-nya bertahan, itu nyata. Kalau menguap, itu tadi cuma noise.

Contoh nyatanya: depth 3 memberi +0.0077 di split seed 42, lalu diverifikasi ulang di seed 2026 dan masih memberi +0.0056. Konsisten, jadi gain-nya nyata. Verifikasi kedua itu yang membedakan temuan dari kebetulan, dan biayanya cuma satu kali menjalankan ulang.

## Urutan kerja dan ekspektasi

Kalau waktunya terbatas, kerjakan berurutan dan berhenti kapan saja — urutannya sudah disusun dari yang paling berdampak.

1. **Bangun harness evaluasi tetap.** Tanpa ini semua eksperimen berikutnya tidak bisa dipercaya. Ini prasyarat, bukan langkah pertama yang bisa dilewati.
2. **Jalankan diagnostik kapasitas.** Tiga model, lima belas menit, menentukan arah seluruh pencarian.
3. **Sapu ulang `CAT_PARAMS`** dengan `depth` direntangkan ke 2–6, maksimal 50 trial, objective-nya **macro-F1 setelah tuning bobot** — bukan CV biasa, karena itu yang benar-benar jadi metrik akhir kalian.
4. **Masukkan `AUTO_CLASS_WEIGHTS` ke ruang pencarian**, termasuk nilai `None`.
5. **Sapu knob aturan keputusan** (`top_frac`, rentang `W_GRID`, `MIN_GAIN`).
6. **Perlakukan keputusan pembersihan sebagai hyperparameter** — pembagi screen time, strategi imputasi.
7. **Verifikasi pemenang akhir di split CV baru** sebelum dipakai sebagai submission final.

Langkah 6 kemungkinan besar tidak menaikkan skor — di pengukuran saya kontribusinya nol. Tetap kerjakan, karena penanganan data kotor masuk kriteria penilaian, dan karena "kami menguji tiga alternatif koreksi dan CV memilih yang ini" jauh lebih kuat di depan juri daripada "kami membaginya 10".

### Soal ekspektasi

Jangan berharap lompatan besar. Setelah sekitar 40 konfigurasi diuji, gain terbaik yang bisa saya dapat dari baseline kalian adalah **+0.006 sampai +0.008** macro-F1. Itu nyata dan terverifikasi, tapi besarnya hanya sekitar sepertiga dari satu standar deviasi noise leaderboard.

Dan itu bukan kegagalan kalian. Tujuh tim teratas di leaderboard terpaut hanya 0.0187 — seluruhnya di dalam rentang noise metrik ini. Satu tim mendapat 0.66108 dengan **satu kali submission**. Artinya semua peserta sudah menemukan sinyal yang sama, dan sisa perbedaannya sebagian besar keberuntungan sampling.

Pelajaran yang paling berharga dari kompetisi ini justru bukan teknik modelnya, tapi kemampuan **mengenali kapan sebuah masalah sudah mentok** — dan tidak menghabiskan dua minggu mengejar 0.005 yang tidak bisa dibedakan dari kebetulan. Kemampuan itu jauh lebih berguna di pekerjaan nyata daripada hafal daftar hyperparameter.

Tenaga yang tersisa lebih baik masuk ke Technical Report dan persiapan presentasi. Di sana selisihnya ditentukan oleh kualitas kerja, bukan oleh lemparan koin.
