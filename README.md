# 🚀 AI Product Success Predictor & Strategic Intelligence Studio

Bu proje, kullanıcının **dilediği ürün veri setini (CSV, Excel, JSON, ZIP)** yükleyebileceği, otomatik veri analizi ve ön işleme yapabilen, çoklu makine öğrenmesi algoritmalarıyla (AutoML) başarı tahmini modeli eğiten, interaktif tekli/toplu ürün başarı tahmini ve **"What-If" simülasyonu** sunan, **e-posta/şifre ile korumalı, çoklu LLM sağlayıcı** (Google Gemini, Anthropic Claude, DeepSeek, Qwen) bağlantılı, **Türkçe / İngilizce / Almanca** arayüze sahip modern bir uçtan uca karar destek sistemidir.

---

## 🌟 Temel Özellikler

1. **🔐 Hesap Girişi (E-posta + Şifre)**:
   - Uygulama açılışında basit ama güvenli bir kayıt/giriş ekranı karşılar.
   - Şifreler asla düz metin olarak saklanmaz — PBKDF2-HMAC-SHA256 (200.000 iterasyon) ile hashlenip yerel bir dosyada (`.local_users.json`, git'e asla dahil edilmez) tutulur.
   - Ek bir veritabanı veya servis gerekmez; tek sunuculu / kurumsal-içi kullanım için tasarlanmıştır.

2. **🌐 Çoklu Dil & Tema**:
   - Arayüz tamamen **Türkçe, İngilizce ve Almanca** olarak kullanılabilir; sol üstten anında değiştirilir.
   - **Açık / Koyu** tema arasında geçiş yapılabilir; tüm grafikler (Plotly) ve kartlar seçilen temaya göre otomatik uyum sağlar.
   - Uygulama ilk açıldığında kısa bir **"Nasıl Kullanılır"** kılavuzu (6 adım) gösterilir.

3. **📁 Evrensel Veri Seti Entegrasyonu**:
   - Dilediğiniz e-ticaret, SaaS, mobil uygulama, online kurs veya kitle fonlaması veri setini (**CSV, XLSX, XLS, JSON ve ZIP Arşivleri**) sürükleyip bırakarak yükleyin.
   - **Otomatik ZIP Ayrıştırma**, dahili **5 hazır sektörel veri seti** (Amazon E-Ticaret, SaaS Lansmanı, Kickstarter, Mobil Uygulama, Online Kurs).
   - Otomatik sütun türü tespiti ve esnek/kural tabanlı hedef değişken türetme.

4. **📊 Otomatik Keşifsel Veri Analizi (Auto EDA) + Veri Sızıntısı Kontrolü**:
   - Veri sağlığı, eksik veri, tekrar eden kayıt analizleri ve Plotly etkileşimli grafikler.
   - **🚨 Otomatik Veri Sızıntısı (Data Leakage) Tespiti**: Hedefle birebir aynı, aşırı yüksek korelasyonlu veya neredeyse tek bir sınıfa işaret eden şüpheli sütunları otomatik tespit edip uyarır — hangi sütunun neden riskli olduğunu ve ne yapılması gerektiğini açıklar.
   - **🕳️ Veri Açıkları Raporu**: Yüksek oranda eksik veya neredeyse hiç değişkenlik göstermeyen sütunları ayrıca listeler.

5. **⚙️ Dinamik Ön İşleme & Özellik Mühendisliği**, **🚀 AutoML Model Eğitimi & Kıyaslama**, **🔮 Canlı Tahmin & "What-If" Simülasyonu**: (Faz 1-2 mirası, değişmedi — bkz. aşağıdaki mimari bölümü.)
   - Ürün fotoğrafı girişi sadeleştirildi: birincil yöntem tek tıkla **sürükle-bırak yükleme**; URL girme isteğe bağlı, katlanır bir bölümde saklı.

6. **🤖 Çoklu LLM AI Ürün Danışmanı & Şematik Görselleştirme**:
   - Tek bir kutucuktan **Google Gemini, Anthropic Claude, DeepSeek veya Qwen** arasında seçim yapıp kendi API anahtarınızı girebilirsiniz.
   - Hiçbir anahtar girilmezse, üç dilde de çalışan **akıllı dahili Heuristic AI Danışman** devreye girer — hiçbir özellik API anahtarı olmadan devre dışı kalmaz.
   - Yönetici Özeti, kişiselleştirilmiş SWOT, somut aksiyon tavsiyeleri, 90 günlük büyüme yol haritası.
   - **🗺️ Önerilerin Şematik Görünümü**: En etkili karar faktörlerinden tahmine, tahminden aksiyon kategorilerine (Fiyatlandırma / İçerik / Pazarlama / Güven) giden bir **Sankey diyagramı**, artı görsel bir **90 günlük yol haritası zaman çizelgesi**.
   - Tek tıkla PDF veya Markdown formatında profesyonel rapor indirme.

7. **🖼️ Görsel Özellikler (Opsiyonel — ResNet50 + PCA)**: Faz 2 mirası; eğitim ile tahmin arasında her zaman gerçek fotoğraf işlenir, sahte/sıfır bir görsel girdi asla kullanılmaz.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Gereksinimlerin Yüklenmesi
```bash
python -m pip install -r requirements.txt
```
Yeni özellikler (çoklu dil, tema, giriş sistemi, çoklu LLM) **hiçbir ek kütüphane gerektirmez** — hepsi mevcut bağımlılıklarla (ve Python'un standart kütüphanesiyle) çalışır.

### 2. (Opsiyonel) LLM API Anahtarları
`.env.example` dosyasını `.env` olarak kopyalayıp istediğiniz sağlayıcıların anahtarını girebilir, ya da arayüzün sol menüsünden doğrudan girebilirsiniz:
```env
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
QWEN_API_KEY=...
```
*(Hiçbiri girilmezse sistem otomatik olarak akıllı yerel AI motorunu devreye alır.)*

### 3. Uygulamayı Başlatma
```bash
streamlit run app.py
```
Uygulama tarayıcınızda `http://localhost:8501` adresinde açılacak ve önce giriş/kayıt ekranını gösterecektir. İlk kullanımda "Hesap Oluştur" sekmesinden ücretsiz bir hesap açabilirsiniz.

---

## 🧪 Testleri Çalıştırma
Uygulamanın tüm pipeline, model eğitimi, giriş sistemi, çoklu dil ve AI danışman bileşenlerini test etmek için:
```bash
python tests/test_pipeline.py
```

---

## 📁 Proje Dizin Yapısı

```
ai-product-success-predictor/
├── app.py                      # Streamlit interaktif web uygulaması
├── requirements.txt            # Python kütüphane bağımlılıkları
├── .env.example                # Örnek çevre değişkenleri (tüm LLM sağlayıcıları)
├── README.md                   # Proje tanıtım ve kullanım kılavuzu
│
├── src/
│   ├── core/
│   │   ├── data_processor.py   # Veri yükleme, sütun tipi tespiti, hedef türetme,
│   │   │                       # veri sızıntısı (leakage) ve veri açığı (gap) tespiti
│   │   ├── eda.py              # Otomatik EDA istatistikleri ve Plotly grafikleri (tema-uyumlu)
│   │   ├── pipeline.py         # Sklearn ColumnTransformer, TF-IDF, One-Hot, Scaling, Görsel özellikler
│   │   ├── image_features.py   # Opsiyonel: ResNet50 + PCA görsel özellik çıkarımı
│   │   ├── models.py           # AutoML model eğitimi, kıyaslama, kayıt
│   │   ├── explainer.py        # Özellik önemi, tekli tahmin sürücüleri, What-If simülatörü
│   │   └── schematics.py       # Önerilerin Sankey diyagramı + 90 günlük yol haritası
│   │
│   ├── llm/
│   │   ├── providers.py        # Gemini / Claude / DeepSeek / Qwen için birleşik LLM istemcisi
│   │   ├── advisor.py          # Çoklu sağlayıcı + akıllı yerel danışman motoru (3 dilde)
│   │   └── prompts.py          # SWOT, strateji ve büyüme prompt şablonları (tr/en/de)
│   │
│   └── utils/
│       ├── sample_data.py      # 5 sektörel örnek veri seti üreticisi
│       ├── report_export.py    # PDF ve Markdown rapor üretici
│       ├── i18n.py             # Türkçe/İngilizce/Almanca çeviri katmanı
│       ├── theme.py            # Açık/Koyu tema CSS'i
│       └── auth.py             # E-posta + şifre kayıt/giriş (PBKDF2 hashleme)
│
└── tests/
    └── test_pipeline.py        # Kapsamlı birim ve entegrasyon testleri (23 test)
```

---

## 🔒 Güvenlik & Üretim Notları

- Şifreler PBKDF2-HMAC-SHA256 (200.000 iterasyon, kullanıcı başına rastgele salt) ile hashlenir; hiçbir zaman düz metin olarak saklanmaz veya loglanmaz.
- Kullanıcı hesapları tek bir yerel JSON dosyasında tutulur — tek sunuculu / kurumsal-içi kullanım için uygundur. Çoklu sunucu / yüksek trafik senaryosunda `src/utils/auth.py` içindeki `JSONUserStore` sınıfının yerine gerçek bir veritabanı (PostgreSQL vb.) bağlanması önerilir; arayüz (`get_user`/`save_user`) bunun için hazırdır.
- "Şifremi unuttum" e-posta akışı şu an dahil değildir (bir SMTP/e-posta servisi gerektirir) — istenirse `JSONUserStore` üzerine eklenebilir.
- API anahtarları yalnızca oturum belleğinde (`st.session_state`) tutulur, diske yazılmaz.
