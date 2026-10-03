# 🚀 AI Product Success Predictor — Kullanım Kılavuzu

Bu belge, uygulamanın ne işe yaradığını ve her bir bölümünün ne için / nasıl kullanıldığını adım adım anlatır.

---

## 1. Bu uygulama ne işe yarar?

**AI Product Success Predictor**, elinizdeki herhangi bir ürün veri setini (Amazon ürünleri, SaaS lansmanları, Kickstarter kampanyaları, mobil uygulamalar, online kurslar, ya da kendi Trendyol/e-ticaret verileriniz) kullanarak:

1. Veriyi otomatik olarak analiz eder ve sağlığını (eksik veri, veri sızıntısı riski vb.) kontrol eder,
2. Birden fazla makine öğrenmesi algoritmasını (AutoML) aynı anda eğitip kıyaslar,
3. Yeni/hayali bir ürünün **başarı olasılığını** (ya da sürekli bir hedefteyse tahmini skorunu) canlı olarak hesaplar,
4. "What-If" simülasyonuyla *"fiyatı düşürsem ne olur?"* gibi senaryoları anında gösterir,
5. Seçtiğiniz bir yapay zeka sağlayıcısı (Gemini / Claude / DeepSeek / Qwen) ya da API anahtarı girilmediğinde dahili akıllı danışman ile **stratejik SWOT analizi ve 90 günlük büyüme yol haritası** üretir,
6. Bunların hepsini PDF/Markdown rapor olarak dışa aktarır.

Kısacası: **"Bu ürünü bu haliyle piyasaya sürersem başarılı olur mu, olmazsa neyi değiştirmeliyim?"** sorusuna veri ve yapay zeka destekli bir cevap verir.

---

## 2. Giriş Ekranı

Uygulama açıldığında önce bir **Giriş / Hesap Oluştur** ekranı gelir. Şifreniz asla düz metin olarak saklanmaz (PBKDF2-HMAC-SHA256 ile hashlenir). İlk kullanımda "Hesap Oluştur" sekmesinden ücretsiz bir hesap açmanız yeterli. Hesabınıza bağlı olarak **tema, renk paleti ve dil tercihiniz** otomatik olarak hatırlanır — bir sonraki girişinizde arayüz kaldığınız yerden, sizin seçtiğiniz görünümle açılır.

---

## 3. Sol Menü (Kenar Çubuğu)

### 3.1. Dil / Görünüm / Renk Paleti (en üstte, giriş yapmadan önce bile görünür)
- **🌐 Dil**: Türkçe, İngilizce, Almanca arasında anında geçiş. Daha fazla dil için aşağıdaki "Daha Fazla Dil" bölümüne bakın (madde 3.3).
- **🎨 Görünüm**: Koyu / Açık tema. Artık gerçekten tüm arayüzü (arka plan, kenar menü, butonlar, giriş kutuları, tablolar) değiştiriyor.
- **🎨 Renk Paleti**: Altı hazır renk teması arasından seçim yapabilirsiniz (İndigo & Mor, Zümrüt & Turkuaz, Okyanus Mavisi, Gün Batımı, Altın & Kehribar, Bektaşi Üzümü). **"🎲 Otomatik"** seçeneği, her girişinizde uygulamanın kendini yeni bir renkle yenilemesini sağlar; istediğiniz an "🔄 Şimdi Yenile" butonuyla da elle tazeleyebilirsiniz. Seçtiğiniz palet hesabınıza kaydedilir.

### 3.2. Veri Seti & Model Kontrolü
- Hazır 5 sektörel veri setinden birini yükleyebilir, ya da kendi CSV/Excel/JSON/ZIP dosyanızı sürükleyip bırakabilirsiniz.

### 3.3. Yapay Zeka (LLM) Ayarları
- **AI Sağlayıcı Seçin**: Gemini, Claude, DeepSeek veya Qwen arasından seçim yapın.
- **API Anahtarı**: İsterseniz kendi anahtarınızı girin (canlı, kişiselleştirilmiş analiz için); girmezseniz dahili akıllı danışman devreye girer — hiçbir özellik anahtarsız devre dışı kalmaz.
- **⚙️ Gelişmiş ayar — Model adı**: Bu kutuya normalde dokunmanıza gerek yok. Sadece sağlayıcının *farklı bir model sürümünü* özellikle denemek isterseniz (ör. daha güçlü/daha ucuz bir model), o modelin tam adını buraya yazarsınız. Boş bırakırsanız sağlayıcının önerdiği varsayılan model otomatik kullanılır.
- **🌍 Daha Fazla Dil (AI Destekli)**: Türkçe/İngilizce/Almanca dışında **onlarca dil** arasından seçim yapabilirsiniz (İspanyolca, Arapça, Rusça, Çince, Fransızca...). Bir API anahtarı girdiyseniz, seçtiğiniz dil o anda (bir kereye mahsus) çevrilir ve önbelleğe alınır — bir sonraki kullanımda anında açılır. Anahtar girilmemişse bu özellik kullanılamaz (arayüz Türkçe/İngilizce/Almanca ile sınırlı kalır), çünkü ~250 arayüz metnini elle her dile çevirmek pratik değil; bunun yerine zaten bağlı olduğunuz yapay zekayı kendi arayüzünü çevirmek için kullanıyoruz.

---

## 4. Sekme 1 — Veri Seti & Hedef Seçimi
Yüklediğiniz verinin önizlemesini, sütun tiplerini (sayısal/kategorik/metin/boolean) ve **tahmin edilecek hedef sütunu** seçtiğiniz yer. Hedef sütun listesi, dosyanızdaki gerçek sütun adlarını gösterir (okunabilirlik için boşluklu/düzgün yazılmış hâlde) — bu isimler sizin dosyanızdan geldiği için çeviremiyoruz, ama daha temiz görünmesini sağladık. Kuralınıza göre kendi özel başarı hedefinizi de burada türetebilirsiniz (ör. "puanı 4.5 üzerinde olanlar başarılı").

## 5. Sekme 2 — Otomatik Keşifsel Analiz (EDA)
Veri setinizin sağlık raporu: kaç satır/sütun, eksik veri oranı, hedef değişkenin dağılımı, **korelasyon ısı haritası** (artık kırmızı-mavi geçişli, pozitif/negatif ilişkiyi renkle net gösteriyor), ve en kritik olarak **🚨 Veri Sızıntısı Tespiti** — hedefle "hile" yapan (yani gerçek hayatta önceden bilinemeyecek) sütunları otomatik yakalayıp size neden riskli olduğunu açıklar.

## 6. Sekme 3 — Ön İşleme & Pipeline
Modelin veriyi nasıl işleyeceğini (ölçekleme yöntemi, kategori sayısı sınırı, metin özellik sayısı) ayarladığınız yer. Ürün fotoğrafı sütununuz varsa, görsellerden otomatik özellik çıkarımını (isteğe bağlı) burada açabilirsiniz.

## 7. Sekme 4 — AutoML Model Eğitimi
"🆕 Yeni Ürün / Lansman Öncesi" kutusunu işaretlerseniz, model sadece lansmandan ÖNCE bilinebilecek özelliklerle eğitilir (yorum sayısı, mevcut satış gibi "sahip olunamayacak" veriler otomatik hariç tutulur) — gerçekçi bir tahmin için önemlidir. Eğitim bitince **kıyaslama tablosu** (artık anlaşılır Türkçe metrik başlıklarıyla: Doğruluk, F1, Kesinlik...) ve **en etkili özellikler** grafiği görünür.

## 8. Sekme 5 — Canlı Tahmin & What-If Simülatörü
Tek bir ürün için tüm alanları doldurup (ya da "🎲 Rastgele Doldur" ile örnek veri çekip) **Tahmin Et**'e basınca: başarı olasılığı göstergesi, en etkili karar faktörleri (artık "category_Elektronik" gibi kodlanmış isimler yerine "Kategori: Elektronik" gibi düz cümlelerle) ve **What-If** ile "bu değişkeni değiştirirsem sonuç nasıl değişir?" simülasyonu görünür. Toplu tahmin için bir CSV yükleyip hepsini aynı anda skorlayabilirsiniz.

## 9. Sekme 6 — LLM Ürün Danışmanı & Rapor
Tahmin ettiğiniz ürün için **Yönetici Özeti, SWOT analizi, somut aksiyon önerileri ve 90 günlük büyüme yol haritasını** (seçtiğiniz dilde) üretir; ayrıca en etkili faktörlerden aksiyon kategorilerine giden bir **Sankey şeması** ve zaman çizelgesi gösterir — hepsi seçtiğiniz renk paletiyle uyumlu. Sonunda tek tıkla **PDF veya Markdown rapor** indirebilirsiniz (regresyon tipi bir tahminde olasılık yerine "bu bir regresyon tahminidir" açıklaması gösterilir, hata vermez).

---

## 10. Bu Turda Neler Düzeltildi / Eklendi?
- Koyu/Açık tema artık gerçekten tüm arayüzü değiştiriyor (önceden sadece birkaç kutucuğu etkiliyordu).
- Metinlerdeki okunma bozukluğu (aşırı geniş bir CSS kuralından kaynaklanıyordu) giderildi.
- Korelasyon ısı haritası artık kırmızı-mavi geçişli (Excel koşullu biçimlendirme gibi).
- Teknik/kodlanmış özellik isimleri ("category_Elektronik" gibi) artık düz Türkçe cümlelere çevriliyor.
- "Model adı" kutusuna açık bir açıklama ve örnek eklendi.
- Regresyon tahminlerinde rapor oluştururken oluşan çökme (hata) düzeltildi.
- **Yeni**: 6 renk paleti + "Otomatik/Kendini Yenile" modu, hesabınıza kaydediliyor.
- **Yeni**: Türkçe/İngilizce/Almanca dışında onlarca dil, yapay zeka ile anlık çevrilip kullanılabiliyor.
