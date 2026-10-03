"""
Internationalization (i18n) Module
Provides a lightweight translation layer for the Streamlit UI.

Two tiers of language support:
1. NATIVE (tr/en/de) — hand-translated, always available, no API key
   needed, used for the login screen and the local heuristic AI advisor.
2. EXTENDED (~40 more world languages) — generated on demand by whichever
   LLM provider the user has already configured (Gemini/Claude/DeepSeek/
   Qwen), translated once and cached to disk so it never needs to be
   re-generated. This is the honest answer to "add every language in the
   world": hand-translating ~250 UI strings per language does not scale,
   but the app already talks to a general-purpose LLM for the AI advisor,
   so reusing that same connection to translate its own interface is a
   natural fit rather than a bolted-on extra.

Usage:
    from src.utils.i18n import t, LANGUAGES
    st.markdown(t("app_title"))
    st.button(t("btn_train", n=5))
"""

from typing import Any, Dict, Optional
import json
import os
import re
import streamlit as st

LANGUAGES = {
    "tr": "🇹🇷 Türkçe",
    "en": "🇬🇧 English",
    "de": "🇩🇪 Deutsch",
}

# ---------------------------------------------------------------------------
# Extended languages — display names only (no hand translation exists for
# these; their UI text is produced by `ensure_dynamic_language` below, on
# demand, via whichever LLM provider/key the user has configured).
# ---------------------------------------------------------------------------
EXTENDED_LANGUAGES: Dict[str, str] = {
    "es": "🇪🇸 Español", "fr": "🇫🇷 Français", "it": "🇮🇹 Italiano", "pt": "🇵🇹 Português",
    "nl": "🇳🇱 Nederlands", "pl": "🇵🇱 Polski", "ru": "🇷🇺 Русский", "uk": "🇺🇦 Українська",
    "ar": "🇸🇦 العربية", "fa": "🇮🇷 فارسی", "he": "🇮🇱 עברית", "hi": "🇮🇳 हिन्दी",
    "ur": "🇵🇰 اردو", "bn": "🇧🇩 বাংলা", "id": "🇮🇩 Bahasa Indonesia", "ms": "🇲🇾 Bahasa Melayu",
    "vi": "🇻🇳 Tiếng Việt", "th": "🇹🇭 ไทย", "zh": "🇨🇳 中文", "ja": "🇯🇵 日本語",
    "ko": "🇰🇷 한국어", "el": "🇬🇷 Ελληνικά", "sv": "🇸🇪 Svenska", "no": "🇳🇴 Norsk",
    "da": "🇩🇰 Dansk", "fi": "🇫🇮 Suomi", "cs": "🇨🇿 Čeština", "sk": "🇸🇰 Slovenčina",
    "hu": "🇭🇺 Magyar", "ro": "🇷🇴 Română", "bg": "🇧🇬 Български", "sr": "🇷🇸 Српски",
    "hr": "🇭🇷 Hrvatski", "az": "🇦🇿 Azərbaycan dili", "kk": "🇰🇿 Қазақша", "ka": "🇬🇪 ქართული",
    "sw": "🇰🇪 Kiswahili", "am": "🇪🇹 አማርኛ", "ne": "🇳🇵 नेपाली", "km": "🇰🇭 ខ្មែរ",
}

# Plain English names, used only inside the LLM instruction that asks a
# live model to answer in this language — not shown in the UI.
LANGUAGE_NAMES_EN: Dict[str, str] = {
    "tr": "Turkish", "en": "English", "de": "German",
    "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "pl": "Polish", "ru": "Russian", "uk": "Ukrainian",
    "ar": "Arabic", "fa": "Persian", "he": "Hebrew", "hi": "Hindi",
    "ur": "Urdu", "bn": "Bengali", "id": "Indonesian", "ms": "Malay",
    "vi": "Vietnamese", "th": "Thai", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "el": "Greek", "sv": "Swedish", "no": "Norwegian",
    "da": "Danish", "fi": "Finnish", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "sr": "Serbian",
    "hr": "Croatian", "az": "Azerbaijani", "kk": "Kazakh", "ka": "Georgian",
    "sw": "Swahili", "am": "Amharic", "ne": "Nepali", "km": "Khmer",
}

ALL_LANGUAGES: Dict[str, str] = {**LANGUAGES, **EXTENDED_LANGUAGES}

DEFAULT_LANG = "tr"

_I18N_CACHE_DIR = os.getenv("APP_I18N_CACHE_DIR", ".i18n_cache")
# In-process memory cache so a dynamically-generated language doesn't need a
# disk read on every single t() call within the same run.
_dynamic_cache: Dict[str, Dict[str, str]] = {}

# ---------------------------------------------------------------------------
# Translation table. Keep keys identical across all three languages.
# Missing keys fall back to Turkish, then to the raw key itself.
# ---------------------------------------------------------------------------
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "tr": {
        # ---- Genel / Header ----
        "app_title": "🚀 AI Product Success Predictor & Intelligence Studio",
        "app_subtitle": "Dilediğiniz veri setini yükleyin, AutoML ile eğitin, anlık başarı olasılığını tahmin edin ve LLM destekli stratejik ürün danışmanlığı alın.",
        "footer_badge": "📌 AI Product Predictor v2.0",
        "footer_caption": "AutoML + Sklearn + XGBoost + LightGBM + Çoklu LLM Sağlayıcı",
        "language_label": "🌐 Dil",
        "theme_label": "🎨 Görünüm",
        "theme_dark": "🌙 Koyu",
        "theme_light": "☀️ Açık",
        "logout_btn": "🚪 Çıkış Yap",
        "logged_in_as": "Giriş yapan: **{name}**",

        # ---- Kimlik doğrulama ----
        "auth_app_tagline": "Ürün başarısını tahmin etmeden önce hesabınıza giriş yapın.",
        "auth_login_tab": "Giriş Yap",
        "auth_register_tab": "Hesap Oluştur",
        "auth_name_label": "Ad Soyad",
        "auth_email_label": "E-posta",
        "auth_password_label": "Şifre",
        "auth_password_confirm_label": "Şifre (Tekrar)",
        "auth_login_btn": "🔓 Giriş Yap",
        "auth_register_btn": "✨ Hesap Oluştur",
        "auth_login_success": "Giriş başarılı! Yönlendiriliyorsunuz…",
        "auth_login_fail": "❌ E-posta veya şifre hatalı.",
        "auth_register_success": "🎉 Hesabınız oluşturuldu! Şimdi giriş yapabilirsiniz.",
        "auth_register_exists": "⚠️ Bu e-posta adresi zaten kayıtlı.",
        "auth_register_mismatch": "⚠️ Şifreler birbiriyle eşleşmiyor.",
        "auth_register_weak": "⚠️ Şifre en az 6 karakter olmalı.",
        "auth_register_invalid_email": "⚠️ Geçerli bir e-posta adresi girin.",
        "auth_required_fields": "⚠️ Lütfen tüm alanları doldurun.",
        "auth_demo_hint": "İlk kez mi kullanıyorsunuz? '{tab}' sekmesinden ücretsiz bir hesap oluşturabilirsiniz. Bilgiler yalnızca bu bilgisayarda, yerel olarak şifrelenmiş biçimde saklanır.",

        # ---- Onboarding kılavuzu ----
        "guide_expander_title": "👋 Yeni misiniz? Uygulamanın nasıl kullanılacağını 1 dakikada öğrenin",
        "guide_intro": "Bu araç 6 adımda çalışır — sırasıyla ilerlemeniz yeterli:",
        "guide_step1": "**1. Veri Seti & Hedef** — Hazır bir sektörel veri seti seçin ya da kendi dosyanızı (CSV/Excel/JSON/ZIP) yükleyin, tahmin etmek istediğiniz sütunu (örn. 'başarılı mı?') seçin.",
        "guide_step2": "**2. Otomatik Analiz (EDA)** — Verinizin sağlığını, eksik değerleri ve hangi özelliklerin başarıyla ilişkili olduğunu otomatik grafiklerle inceleyin.",
        "guide_step3": "**3. Ön İşleme** — Varsayılan ayarlar çoğu veri seti için yeterlidir; isterseniz ince ayar yapın. Ürün fotoğrafı varsa burada etkinleştirin.",
        "guide_step4": "**4. Model Eğitimi** — Tek tıkla birden fazla makine öğrenmesi modelini eğitip karşılaştırın; en başarılı model otomatik seçilir.",
        "guide_step5": "**5. Canlı Tahmin & What-If** — Yeni bir ürünün bilgilerini girin, başarı olasılığını anında görün ve 'ne olursa' senaryolarını deneyin.",
        "guide_step6": "**6. AI Danışman & Rapor** — Yapay zekadan kişiselleştirilmiş SWOT analizi ve büyüme önerileri alın, PDF/Markdown rapor olarak indirin.",
        "guide_tip": "💡 İpucu: Sol menüden istediğiniz zaman dil, görünüm (açık/koyu) ve yapay zeka sağlayıcınızı değiştirebilirsiniz.",

        # ---- Sidebar: Veri kaynağı ----
        "sidebar_data_control_header": "⚙️ Veri & Model Kontrolü",
        "dataset_source_label": "Veri Seti Kaynağı:",
        "dataset_source_builtin": "Hazır Sektörel Veri Setleri",
        "dataset_source_custom": "Kendi Veri Setini Yükle (CSV/Excel/JSON)",
        "sector_select_label": "Sektör Seçin:",
        "load_dataset_btn": "🔄 Veri Setini Yükle",
        "dataset_loaded_msg": "{name} yüklendi!",
        "upload_file_label": "Veri Dosyanızı Yükleyin:",
        "upload_file_help": "Sürükleyip bırakabilir veya dosya seçebilirsiniz (ZIP arşivleri desteklenir).",
        "zip_no_valid_file": "⚠️ Yüklenen ZIP arşivi içinde geçerli bir veri dosyası (CSV/Excel/JSON) bulunamadı.",
        "zip_single_info": "📦 ZIP Arşivi: `{filename}` ({size})",
        "zip_multi_found": "📦 **ZIP içinde {n} veri dosyası bulundu:**",
        "zip_choose_file": "Yüklenecek Dosyayı Seçin:",
        "zip_read_error": "ZIP arşivi okunurken hata: {error}",
        "process_upload_btn": "📥 Yüklenen Veriyi İşle",
        "upload_success": "'{name}' başarıyla yüklendi! ({n:,} satır)",
        "upload_error": "Veri yüklenirken hata oluştu: {error}",

        # ---- Sidebar: LLM ----
        "llm_settings_header": "🤖 Yapay Zeka (LLM) Ayarları",
        "llm_provider_label": "AI Sağlayıcı Seçin:",
        "llm_api_key_label": "{provider} API Anahtarı (Opsiyonel):",
        "llm_api_key_help": "Anahtar girildiğinde canlı {provider} modeli kullanılır. Girilmezse akıllı dahili AI danışmanı devreye girer.",
        "llm_active_live": "🟢 Canlı {provider} API aktif.",
        "llm_active_heuristic": "🔵 Akıllı Dahili Heuristic AI Motoru aktif.",
        "llm_model_override_label": "Model adı (opsiyonel, boş bırakılabilir):",
        "llm_model_override_expander_title": "⚙️ Gelişmiş ayar: Belirli bir model sürümü kullan (opsiyonel)",
        "llm_model_override_help": (
            "Bu kutu, seçtiğiniz sağlayıcının **hangi model sürümüyle** çalışacağını belirler — "
            "çoğu kullanıcının dokunmasına gerek yoktur. Boş bırakırsanız sağlayıcının önerilen "
            "varsayılan modeli (**{default_model}**) otomatik kullanılır. Sadece farklı/daha yeni bir "
            "model sürümünü özellikle denemek isterseniz, sağlayıcının belgelerinde geçen tam model "
            "adını buraya yazabilirsiniz (örn. {example})."
        ),

        # ---- Genişletilmiş Diller (AI destekli anlık çeviri) ----
        "active_extended_lang_note": "🌍 Aktif dil: **{lang}** (AI ile üretildi)",
        "extended_lang_header": "Daha Fazla Dil (AI Destekli)",
        "extended_lang_help": (
            "Türkçe/İngilizce/Almanca dışında **dünya genelinde onlarca dil** arasından seçim "
            "yapabilirsiniz — arayüz metinleri, yukarıda seçtiğiniz AI sağlayıcısı kullanılarak "
            "bir kereye mahsus çevrilir ve önbelleğe alınır; sonraki kullanımlar anında olur. "
            "Bu nedenle bir API anahtarı girilmiş olması gerekir."
        ),
        "extended_lang_select_label": "Dil seçin:",
        "extended_lang_apply_btn": "🌍 Bu Dile Geç",
        "extended_lang_need_key": "Bu dili kullanmak için önce yukarıdan bir {provider} API anahtarı girmelisiniz — anahtar girilmeden yalnızca Türkçe/İngilizce/Almanca kullanılabilir.",
        "extended_lang_translating": "{lang} diline çevriliyor (bu sadece ilk seferde olur)...",
        "extended_lang_success": "✅ Arayüz artık {lang} dilinde!",
        "extended_lang_fail": "❌ Çeviri oluşturulamadı — API anahtarınızı kontrol edin ve tekrar deneyin.",
        "extended_lang_advisor_note": "ℹ️ AI Danışman (Sekme 6), API anahtarı girildiğinde doğrudan bu dilde yazar. Anahtar girilmezse dahili heuristik danışman yalnızca Türkçe/İngilizce/Almanca destekler ve bu durumda Türkçe'ye döner.",

        # ---- Renk Paleti / "Kendini Yenile" ----
        "accent_label": "🎨 Renk Paleti",
        "accent_auto": "🎲 Otomatik (Her Girişte Yenilenir)",
        "accent_indigo": "İndigo & Mor",
        "accent_emerald": "Zümrüt & Turkuaz",
        "accent_ocean": "Okyanus Mavisi",
        "accent_sunset": "Gün Batımı",
        "accent_gold": "Altın & Kehribar",
        "accent_berry": "Bektaşi Üzümü (Mor & Pembe)",
        "accent_help": "Rengi kendiniz seçebilir ya da 'Otomatik' bırakarak uygulamanın her girişte kendini yeni bir renkle yenilemesini sağlayabilirsiniz. Seçiminiz hesabınıza kaydedilir.",
        "accent_shuffle_btn": "🔄 Şimdi Yenile",
        "accent_saved_note": "💾 Tercihiniz hesabınıza kaydedildi.",

        # ---- Sekmeler ----
        "tab1_title": "📁 1. Veri Seti & Hedef Seçimi",
        "tab2_title": "📊 2. Otomatik Keşifsel Analiz (EDA)",
        "tab3_title": "⚙️ 3. Ön İşleme & Pipeline",
        "tab4_title": "🚀 4. AutoML Model Eğitimi",
        "tab5_title": "🔮 5. Canlı Ürün Tahmini & Simülatör",
        "tab6_title": "🤖 6. LLM Ürün Danışmanı & Rapor",

        # ---- Tab 1 ----
        "t1_subheader": "📁 Veri Seti İnceleme ve Hedef Değişken Belirleme",
        "t1_active_dataset": "Aktif Veri Seti: **{name}** — **{rows:,} satır**, **{cols} sütun**",
        "t1_target_select": "🎯 Tahmin Edilecek Hedef Sütun (Target):",
        "t1_target_select_help": "Bu liste, yüklediğiniz veri dosyasındaki sütun adlarını gösterir — okunabilirlik için boşluklu/baş harfleri büyük hale getirildi, ancak isimler doğrudan sizin dosyanızdan geldiği için (örn. Trendyol veri şemanız) İngilizce sütun adlarını Türkçeye çeviremiyoruz; sütunları yeniden adlandırmak isterseniz kaynak dosyanızdaki başlıkları değiştirip yeniden yükleyebilirsiniz.",
        "t1_derive_expander": "🛠️ Kurala Dayalı Özel Başarı Hedefi Türet (Opsiyonel)",
        "t1_derive_help": "Mevcut bir sayısal sütundan (örn. satış > 500 veya puan >= 4.5) yeni bir başarı/başarısızlık hedefi oluşturabilirsiniz.",
        "t1_derive_base_col": "Hangi Sütun Üzerinden?",
        "t1_derive_condition": "Koşul:",
        "t1_derive_threshold_pct": "Eşik Değer (Üst Yüzde %):",
        "t1_derive_threshold": "Eşik Değer (Threshold):",
        "t1_derive_new_name": "Yeni Hedef Sütun Adı:",
        "t1_derive_btn": "✨ Özel Hedef Sütununu Oluştur ve Seç",
        "t1_derive_success": "'{name}' hedef sütunu başarıyla oluşturuldu ve hedef olarak seçildi!",
        "t1_derive_leak_warning": "⚠️ Veri sızıntısını (data leakage) önlemek için '{col}' sütunu — hedef doğrudan bundan türetildiği için — özellik setinden otomatik olarak çıkarıldı.",
        "t1_task_type_line": "**Görev Türü:** `{task}` | **Alt Tür:** `{subtype}`",
        "t1_task_classification": "Sınıflandırma (Classification)",
        "t1_task_regression": "Regresyon (Regression)",
        "t1_preview_header": "🔍 Veri Önizlemesi (İlk 50 Satır)",
        "t1_column_types_header": "🧩 Otomatik Tespit Edilen Sütun Türleri",
        "t1_numeric": "🔢 Sayısal ({n})",
        "t1_categorical": "🏷️ Kategorik ({n})",
        "t1_text": "📝 Metin ({n})",
        "t1_boolean": "🔘 İkili / Bool ({n})",
        "t1_none": "Yok",

        # ---- Tab 2 ----
        "t2_subheader": "📊 Otomatik Keşifsel Veri Analizi (Auto EDA)",
        "t2_kpi_rows": "Toplam Satır",
        "t2_kpi_cols": "Toplam Özellik",
        "t2_kpi_missing": "Eksik Değerler",
        "t2_kpi_duplicates": "Tekrar Eden Satır",
        "t2_insights_header": "💡 Hızlı İstatistiksel Bulgular",
        "t2_feature_deep_dive": "🔬 Özellik — Hedef İlişkisi Derinlemesine İnceleme",
        "t2_feature_select": "İncelemek İstediğiniz Özellik:",
        "t2_corr_insufficient": "Korelasyon ısı haritası için yeterli sayısal özellik bulunamadı.",
        "t2_leakage_header": "🚨 Veri Sızıntısı (Data Leakage) Kontrolü",
        "t2_leakage_none": "✅ Belirgin bir veri sızıntısı sinyali tespit edilmedi.",
        "t2_leakage_found": "⚠️ Aşağıdaki sütunlar hedefle şüpheli derecede güçlü ilişki gösteriyor — modelin gerçek hayatta çalışmayacak kadar 'kolay' öğrenmesine neden olabilirler:",
        "t2_leakage_row": "**`{col}`** — {reason} *(önerilen aksiyon: {action})*",
        "t2_gap_header": "🕳️ Veri Açıkları & Kalite Notları",
        "t2_gap_high_missing": "**`{col}`** sütununun **%{pct:.1f}**'i eksik — bu oran yüksek, modele dahil edilmeden önce doldurma stratejisini gözden geçirin.",
        "t2_gap_low_variance": "**`{col}`** neredeyse hiç değişkenlik göstermiyor (tek bir değer baskın) — modele ayırt edici bir bilgi katmıyor olabilir.",
        "t2_gap_none": "✅ Belirgin bir veri boşluğu / kalite sorunu bulunamadı.",
        "leak_reason_duplicate_of_target": "hedef sütunla birebir aynı değerlere sahip",
        "leak_reason_high_correlation": "hedefle çok yüksek korelasyon gösteriyor (r={corr})",
        "leak_reason_moderate_correlation": "hedefle orta-yüksek korelasyon gösteriyor (r={corr})",
        "leak_reason_categorical_purity": "'{value}' kategorisi neredeyse tek bir sınıfa işaret ediyor",
        "leak_action_remove_column": "bu sütunu özellik setinden çıkarın",
        "leak_action_remove_or_check": "büyük olasılıkla çıkarılmalı — sütunun tanımını kontrol edin",
        "leak_action_review_column": "sütunun anlamını gözden geçirin, gerekirse çıkarın",

        # ---- Tab 3 ----
        "t3_subheader": "⚙️ Özellik Mühendisliği ve Ön İşleme Hattı",
        "t3_intro": "Veri setinizdeki farklı sütun tipleri için otomatik, bellek dostu akıllı ön işleme hattı yapılandırılır.",
        "t3_scaler": "Sayısal Özellik Ölçekleyici (Scaler):",
        "t3_max_cats": "Maksimum Kategori Sayısı (One-Hot Cap):",
        "t3_max_cats_help": "Bellek taşmalarını önler, nadir kategorileri 'Diğer' altında toplar.",
        "t3_tfidf": "Maksimum TF-IDF Metin Özelliği:",
        "t3_test_split": "Test Verisi Ayrım Oranı (%):",
        "t3_steps_header": "📋 Ön İşleme Adımları Özeti:",
        "t3_image_header": "🖼️ Ürün Fotoğrafı (Opsiyonel)",
        "t3_image_checkbox": "Ürün fotoğraflarından görsel özellik çıkar (ResNet50 + PCA)",
        "t3_image_help": "İlk çalıştırmada fotoğraflar indirilip işlenir (yavaş olabilir); sonraki çalıştırmalarda diskteki önbellek sayesinde hızlanır. Ek kütüphane gerektirir.",
        "t3_image_no_col": "⚠️ Veri setinde görsel URL'si gibi görünen bir sütun bulunamadı; bu seçenek atlanacak.",
        "t3_image_col_select": "Hangi sütun ürün fotoğrafının URL'sini içeriyor?",
        "t3_image_pca_dim": "Görsel PCA Boyutu:",
        "t3_image_sample_n": "PCA İçin Örneklenecek Foto Sayısı:",
        "t3_image_sample_help": "Sadece PCA temelini kurmak için kullanılan örneklem sayısı — eğitim/tahmin sırasında her satırın gerçek fotoğrafı yine de işlenir.",
        "t3_image_note": "ℹ️ Bu sütun artık modele özellik olarak dahil edilecek. Tekli tahmin, 'What-If' simülasyonu ve toplu (batch) skorlama sırasında da her satırın gerçek fotoğrafı o an indirilip işlenir.",

        # ---- Tab 4 ----
        "t4_subheader": "🚀 AutoML Çoklu Model Eğitimi & Kıyaslama",
        "t4_intro": "Algoritmalar aynı eğitim/test ayrımında eğitilir, metrikleri kıyaslanır ve en yüksek başarıya sahip model otomatik seçilir.",
        "t4_model_select": "Eğitilecek Modelleri Seçin:",
        "t4_sample_size_label": "⚡ Büyük Veri Eğitimi Örneklem Boyutu:",
        "t4_sample_size_help": "Mevcut veri seti {n:,} satır. Hızlı sonuç ve bellek güvenliği için dengeli örneklem seçebilirsiniz.",
        "t4_dataset_size_info": "ℹ️ Veri Seti Boyutu: {n:,} satır (Tüm veri seti ile eğitilecek).",
        "t4_new_product_checkbox": "🆕 Bu model henüz piyasaya sürülmemiş / satışı olmayan ürünler için kullanılacak",
        "t4_new_product_help": "İşaretlerseniz, 'reviews', 'rating', 'sales' gibi ürünün MEVCUT çekişini yansıtan sütunlar eğitimden tamamen çıkarılır.",
        "t4_new_product_excluded": "🆕 Yeni ürün modu: {cols} sütunları — mevcut çekiş sinyalleri oldukları için — eğitimden çıkarıldı.",
        "t4_new_product_none_found": "🆕 Yeni ürün modu açık, ancak veri setinde çekiş sinyali gibi görünen bir sütun bulunamadı.",
        "t4_select_warning": "⚠️ Lütfen eğitmek için en az bir model seçin.",
        "t4_train_btn": "🚀 AutoML Model Eğitimini Başlat",
        "t4_training_spinner": "Modeller eğitiliyor, çapraz değerlendirme ve metrikler hesaplanıyor…",
        "t4_train_success": "🎉 Eğitim tamamlandı! En başarılı model: **{name}**",
        "t4_train_image_error": "⚠️ {error}",
        "t4_train_value_error": "⚠️ Eğitim tamamlanamadı: {error}",
        "t4_leakage_train_warning": "🚨 Uyarı: seçtiğiniz özellikler arasında yüksek sızıntı riski taşıyan sütunlar var. Bunlar eğitilen modelin gerçek dünyada olduğundan çok daha 'başarılı' görünmesine yol açabilir — 2. sekmedeki 'Veri Sızıntısı Kontrolü' bölümüne bakın.",
        "t4_leaderboard_header": "🏆 Model Liderlik Tablosu (Leaderboard)",
        "t4_best_model_header": "🌟 En Başarılı Model: `{name}`",
        "t4_fi_missing": "Özellik önemi grafiği bu model tipi için doğrudan çıkarılamadı.",

        # ---- Tab 5 ----
        "t5_subheader": "🔮 Canlı Ürün Başarı Tahmini & 'What-If' Simülasyonu",
        "t5_need_training": "⚠️ Lütfen önce **4. AutoML Model Eğitimi** sekmesinden bir model eğitin!",
        "t5_subtab_single": "🧪 Tekli Ürün Tahmini & Simülasyon",
        "t5_subtab_batch": "📁 Toplu Dosya Tahmini (Batch CSV)",
        "t5_form_intro": "Aşağıdaki formu doldurarak yeni bir ürünün piyasadaki tahmini başarı oranını anında hesaplayın.",
        "t5_traction_note": "ℹ️ {cols} gibi alanlar ürünün **mevcut** performansını yansıtır. Henüz piyasaya sürülmemiş yeni bir ürünü değerlendiriyorsanız bu alanları 0 olarak girin, ya da **4. AutoML** sekmesindeki '🆕 Yeni Ürün' seçeneğiyle modeli yeniden eğitin.",
        "t5_random_fill_btn": "🎲 Rastgele Örnek Ürün Değerleri Doldur",
        "t5_photo_header": "🖼️ Ürün Fotoğrafı",
        "t5_photo_upload_label": "Fotoğrafı sürükleyip bırakın ya da seçin",
        "t5_photo_upload_help": "En kolay yöntem budur — JPG/PNG/WEBP. Fotoğraf tahmin sırasında gerçekten işlenir.",
        "t5_photo_url_expander": "🔗 Bunun yerine bir fotoğraf linki (URL) kullan",
        "t5_photo_url_label": "Fotoğraf URL'si:",
        "t5_photo_preview_caption": "Yüklenen fotoğraf",
        "t5_photo_none_caption": "Henüz fotoğraf eklenmedi",
        "t5_predict_btn": "🔮 Ürün Başarısını Tahmin Et",
        "t5_result_header": "🎯 Tahmin Sonuç Kartı",
        "t5_gauge_title_pct": "Başarı Olasılığı",
        "t5_gauge_title_score": "Tahmin Skoru",
        "t5_level_high": "🟢 YÜKSEK BAŞARI POTANSİYELİ",
        "t5_level_mid": "🟡 ORTA BAŞARI / OPTİMİZASYON ÖNERİLİR",
        "t5_level_low": "🔴 DÜŞÜK BAŞARI RİSKİ / ACİL OPTİMİZASYON",
        "t5_level_neutral": "📊 REGRESYON TAHMİNİ",
        "t5_prob_label": "Başarı Olasılığı",
        "t5_prediction_label": "**Tahmin:** {value}",
        "t5_model_label": "**Model:** `{value}`",
        "t5_proba_note": "ℹ️ Bu model predict_proba desteklemiyor; tahmin sınıfından türetildi.",
        "t5_drivers_header": "**⚖️ En Etkili Faktörler:**",
        "t5_success_label": "✅ Başarılı (1)",
        "t5_failure_label": "❌ Başarısız (0)",
        "t5_whatif_header": "🎛️ 'What-If' Duyarlılık & Parametre Simülatörü",
        "t5_whatif_intro": "Belirlediğiniz bir özelliği değiştirerek ürünün başarı ihtimalinin nasıl tepki verdiğini simüle edin.",
        "t5_whatif_var_select": "Değişken Seçin:",
        "t5_whatif_run_btn": "⚡ Simülasyonu Çalıştır",
        "t5_batch_header": "📁 Toplu Ürün Skorlama (Batch Prediction)",
        "t5_batch_image_note": "🖼️ Görsel özellikler etkin: her satırın fotoğrafı gerçekten indirilip işlenecek, bu yüzden görselli büyük dosyalarda skorlama normalden belirgin şekilde yavaş olabilir.",
        "t5_batch_upload_label": "Tahmin edilecek CSV / Excel / JSON / ZIP dosyasını yükleyin:",
        "t5_batch_score_btn": "🚀 Tüm Ürünleri Skorla",
        "t5_batch_spinner": "Toplu tahminler hesaplanıyor…",
        "t5_batch_success": "{n:,} ürün başarıyla skorlandı!",
        "t5_batch_download_btn": "📥 Skorlanmış Sonuçları İndir (CSV)",

        # ---- Tab 6 ----
        "t6_subheader": "🤖 LLM Ürün Başarı Danışmanı & Stratejik Rapor",
        "t6_live_active": "🟢 **{provider}** bağlı — Gerçek zamanlı LLM analizi aktif.",
        "t6_heuristic_expander": "🔵 Dahili AI Danışmanı aktif — Canlı bir LLM'e geçmek için tıkla",
        "t6_heuristic_body": "Sol panelden bir **AI Sağlayıcı** seçip API anahtarınızı girerek canlı bir LLM modelini aktif edebilirsiniz (Gemini, Claude, DeepSeek veya Qwen).\n\nCanlı LLM etkinleştirildiğinde:\n- ✅ 90 günlük tablo formatında büyüme yol haritası\n- ✅ Ürünün özelliklerine özgü kişiselleştirilmiş SWOT\n- ✅ Quick Wins (Bu hafta yapılabilecek 3 somut aksiyon)\n- ✅ Fiyat & başlık optimizasyonu için somut örnekler",
        "t6_product_summary_header": "📦 Tahmin Yapılan Ürün",
        "t6_product_summary_line": "Model: `{model}` &nbsp;|&nbsp; Tahmin: `{pred}` &nbsp;|&nbsp; {prob}",
        "t6_analyze_btn": "🤖 AI Analizi & Stratejik Önerileri Oluştur",
        "t6_analyze_spinner": "🧠 Yapay zeka analiz yapıyor, stratejik öneriler ve SWOT hazırlanıyor…",
        "t6_need_prediction": "💡 Lütfen **5. Canlı Ürün Tahmini** sekmesinde bir ürün için 'Ürün Başarısını Tahmin Et' butonuna basarak yapay zeka analizini tetikleyin.",
        "t6_engine_label": "**Motor:** `{source}`",
        "t6_reanalyze_btn": "🔄 Yeniden Analiz Et",
        "t6_reset_btn": "🧹 Sıfırla",
        "t6_summary_quote": "> **Ürün Tahmini:** {pred} &nbsp;|&nbsp; {prob} **Model:** `{model}`",
        "t6_schematic_header": "🗺️ Önerilerin Şematik Görünümü",
        "t6_schematic_caption": "En etkili faktörlerden tahmine, tahminden aksiyon kategorilerine giden akışı gösterir.",
        "t6_roadmap_header": "🚀 90 Günlük Yol Haritası (Görsel Zaman Çizelgesi)",
        "t6_roadmap_phase1": "Lansman & Doğrulama",
        "t6_roadmap_phase2": "Ölçeklenme",
        "t6_roadmap_phase3": "Sürdürülebilirlik",
        "t6_export_header": "📥 Raporu Dışa Aktar",
        "t6_export_md_btn": "📄 Markdown Rapor (.md)",
        "t6_export_pdf_btn": "📑 PDF Rapor (.pdf)",

        # ---- Sankey / şema düğümleri ----
        "sankey_node_prediction_success": "Yüksek Başarı",
        "sankey_node_prediction_risk": "Optimizasyon Gerekli",
        "sankey_cat_pricing": "Fiyatlandırma",
        "sankey_cat_content": "Başlık / İçerik",
        "sankey_cat_marketing": "Pazarlama",
        "sankey_cat_trust": "Güven / Sosyal Kanıt",
    },

    "en": {
        "app_title": "🚀 AI Product Success Predictor & Intelligence Studio",
        "app_subtitle": "Upload any dataset, train with AutoML, get instant success probability, and receive LLM-powered strategic product advice.",
        "footer_badge": "📌 AI Product Predictor v2.0",
        "footer_caption": "AutoML + Sklearn + XGBoost + LightGBM + Multi-Provider LLM",
        "language_label": "🌐 Language",
        "theme_label": "🎨 Appearance",
        "theme_dark": "🌙 Dark",
        "theme_light": "☀️ Light",
        "logout_btn": "🚪 Log Out",
        "logged_in_as": "Logged in as: **{name}**",

        "auth_app_tagline": "Sign in to your account before predicting product success.",
        "auth_login_tab": "Sign In",
        "auth_register_tab": "Create Account",
        "auth_name_label": "Full Name",
        "auth_email_label": "Email",
        "auth_password_label": "Password",
        "auth_password_confirm_label": "Confirm Password",
        "auth_login_btn": "🔓 Sign In",
        "auth_register_btn": "✨ Create Account",
        "auth_login_success": "Signed in successfully! Redirecting…",
        "auth_login_fail": "❌ Incorrect email or password.",
        "auth_register_success": "🎉 Account created! You can sign in now.",
        "auth_register_exists": "⚠️ This email is already registered.",
        "auth_register_mismatch": "⚠️ Passwords do not match.",
        "auth_register_weak": "⚠️ Password must be at least 6 characters.",
        "auth_register_invalid_email": "⚠️ Please enter a valid email address.",
        "auth_required_fields": "⚠️ Please fill in all fields.",
        "auth_demo_hint": "First time here? Create a free account from the '{tab}' tab. Credentials are stored locally on this machine, hashed — never in plain text.",

        "guide_expander_title": "👋 New here? Learn how this app works in 1 minute",
        "guide_intro": "This tool works in 6 steps — just follow them in order:",
        "guide_step1": "**1. Dataset & Target** — Pick a built-in industry dataset or upload your own (CSV/Excel/JSON/ZIP), then choose the column you want to predict (e.g. 'is successful?').",
        "guide_step2": "**2. Automated Analysis (EDA)** — Review data health, missing values, and which features correlate with success, through automatic charts.",
        "guide_step3": "**3. Preprocessing** — The defaults work for most datasets; fine-tune if you like. Enable a product photo column here if you have one.",
        "guide_step4": "**4. Model Training** — Train and compare several machine learning models with one click; the best one is picked automatically.",
        "guide_step5": "**5. Live Prediction & What-If** — Enter a new product's details, see its success probability instantly, and try 'what if' scenarios.",
        "guide_step6": "**6. AI Advisor & Report** — Get a personalized SWOT analysis and growth recommendations from AI, and download it as a PDF/Markdown report.",
        "guide_tip": "💡 Tip: you can change the language, appearance (light/dark) and AI provider anytime from the sidebar.",

        "sidebar_data_control_header": "⚙️ Data & Model Controls",
        "dataset_source_label": "Dataset Source:",
        "dataset_source_builtin": "Built-in Industry Datasets",
        "dataset_source_custom": "Upload Your Own Dataset (CSV/Excel/JSON)",
        "sector_select_label": "Choose a Sector:",
        "load_dataset_btn": "🔄 Load Dataset",
        "dataset_loaded_msg": "{name} loaded!",
        "upload_file_label": "Upload Your Data File:",
        "upload_file_help": "Drag & drop or browse (ZIP archives are supported).",
        "zip_no_valid_file": "⚠️ No valid data file (CSV/Excel/JSON) found inside the uploaded ZIP.",
        "zip_single_info": "📦 ZIP Archive: `{filename}` ({size})",
        "zip_multi_found": "📦 **Found {n} data files inside the ZIP:**",
        "zip_choose_file": "Choose the File to Load:",
        "zip_read_error": "Error reading ZIP archive: {error}",
        "process_upload_btn": "📥 Process Uploaded Data",
        "upload_success": "'{name}' loaded successfully! ({n:,} rows)",
        "upload_error": "Error while loading the data: {error}",

        "llm_settings_header": "🤖 AI (LLM) Settings",
        "llm_provider_label": "Choose an AI Provider:",
        "llm_api_key_label": "{provider} API Key (Optional):",
        "llm_api_key_help": "When set, the live {provider} model is used. If left empty, the smart built-in AI advisor takes over.",
        "llm_active_live": "🟢 Live {provider} API active.",
        "llm_active_heuristic": "🔵 Smart built-in heuristic AI engine active.",
        "llm_model_override_label": "Model name (optional, leave blank for default):",
        "llm_model_override_expander_title": "⚙️ Advanced setting: use a specific model version (optional)",
        "llm_model_override_help": (
            "This box controls **which exact model version** of your chosen provider is called — "
            "most people never need to touch it. Leave it blank and the provider's recommended "
            "default model (**{default_model}**) is used automatically. Only fill it in if you "
            "specifically want to try a different/newer model version — enter the exact model name "
            "from the provider's own documentation (e.g. {example})."
        ),

        "active_extended_lang_note": "🌍 Active language: **{lang}** (AI-generated)",
        "extended_lang_header": "More Languages (AI-Powered)",
        "extended_lang_help": (
            "Beyond Turkish/English/German, you can pick from **dozens of languages worldwide** — "
            "the interface text is translated once using the AI provider you set up above, then "
            "cached; every use after that is instant. This is why an API key is required."
        ),
        "extended_lang_select_label": "Choose a language:",
        "extended_lang_apply_btn": "🌍 Switch to This Language",
        "extended_lang_need_key": "You need to enter a {provider} API key above first to use this language — without a key, only Turkish/English/German are available.",
        "extended_lang_translating": "Translating into {lang} (this only happens once)...",
        "extended_lang_success": "✅ The interface is now in {lang}!",
        "extended_lang_fail": "❌ Translation failed — check your API key and try again.",
        "extended_lang_advisor_note": "ℹ️ The AI Advisor (Tab 6) writes directly in this language once an API key is set. Without a key, the built-in heuristic advisor only supports Turkish/English/German and falls back to Turkish.",

        "accent_label": "🎨 Color Palette",
        "accent_auto": "🎲 Auto (Refreshes Every Login)",
        "accent_indigo": "Indigo & Purple",
        "accent_emerald": "Emerald & Teal",
        "accent_ocean": "Ocean Blue",
        "accent_sunset": "Sunset",
        "accent_gold": "Gold & Amber",
        "accent_berry": "Berry (Purple & Pink)",
        "accent_help": "Pick a color yourself, or leave it on 'Auto' so the app refreshes itself with a new color every time you log in. Your choice is saved to your account.",
        "accent_shuffle_btn": "🔄 Refresh Now",
        "accent_saved_note": "💾 Your preference was saved to your account.",

        "tab1_title": "📁 1. Dataset & Target Selection",
        "tab2_title": "📊 2. Automated Exploratory Analysis (EDA)",
        "tab3_title": "⚙️ 3. Preprocessing & Pipeline",
        "tab4_title": "🚀 4. AutoML Model Training",
        "tab5_title": "🔮 5. Live Prediction & Simulator",
        "tab6_title": "🤖 6. LLM Product Advisor & Report",

        "t1_subheader": "📁 Dataset Review and Target Variable Selection",
        "t1_active_dataset": "Active Dataset: **{name}** — **{rows:,} rows**, **{cols} columns**",
        "t1_target_select": "🎯 Target Column to Predict:",
        "t1_target_select_help": "This list shows the column names from the data file you uploaded — spaced/capitalized for readability, but since the names come straight from your own file, we can't translate them into another language; if you want different column names, rename the headers in your source file and re-upload it.",
        "t1_derive_expander": "🛠️ Derive a Custom Success Target from a Rule (Optional)",
        "t1_derive_help": "Create a new success/failure target from an existing numeric column (e.g. sales > 500 or rating >= 4.5).",
        "t1_derive_base_col": "Based on Which Column?",
        "t1_derive_condition": "Condition:",
        "t1_derive_threshold_pct": "Threshold (Top Percentage %):",
        "t1_derive_threshold": "Threshold:",
        "t1_derive_new_name": "New Target Column Name:",
        "t1_derive_btn": "✨ Create and Select Custom Target",
        "t1_derive_success": "'{name}' target column created and selected successfully!",
        "t1_derive_leak_warning": "⚠️ To avoid data leakage, '{col}' was automatically removed from the feature set — because the target is derived directly from it.",
        "t1_task_type_line": "**Task Type:** `{task}` | **Subtype:** `{subtype}`",
        "t1_task_classification": "Classification",
        "t1_task_regression": "Regression",
        "t1_preview_header": "🔍 Data Preview (First 50 Rows)",
        "t1_column_types_header": "🧩 Automatically Detected Column Types",
        "t1_numeric": "🔢 Numeric ({n})",
        "t1_categorical": "🏷️ Categorical ({n})",
        "t1_text": "📝 Text ({n})",
        "t1_boolean": "🔘 Boolean ({n})",
        "t1_none": "None",

        "t2_subheader": "📊 Automated Exploratory Data Analysis (Auto EDA)",
        "t2_kpi_rows": "Total Rows",
        "t2_kpi_cols": "Total Features",
        "t2_kpi_missing": "Missing Values",
        "t2_kpi_duplicates": "Duplicate Rows",
        "t2_insights_header": "💡 Quick Statistical Findings",
        "t2_feature_deep_dive": "🔬 Feature — Target Relationship Deep Dive",
        "t2_feature_select": "Feature to Inspect:",
        "t2_corr_insufficient": "Not enough numeric features for a correlation heatmap.",
        "t2_leakage_header": "🚨 Data Leakage Check",
        "t2_leakage_none": "✅ No obvious data leakage signal detected.",
        "t2_leakage_found": "⚠️ The following columns show a suspiciously strong relationship with the target — they may make the model learn 'too easily' to work in the real world:",
        "t2_leakage_row": "**`{col}`** — {reason} *(suggested action: {action})*",
        "t2_gap_header": "🕳️ Data Gaps & Quality Notes",
        "t2_gap_high_missing": "**`{col}`** is **{pct:.1f}%** missing — that's a high rate; review your imputation strategy before including it in the model.",
        "t2_gap_low_variance": "**`{col}`** shows almost no variation (one value dominates) — it may not add distinguishing information to the model.",
        "t2_gap_none": "✅ No notable data gaps or quality issues found.",
        "leak_reason_duplicate_of_target": "has exactly the same values as the target column",
        "leak_reason_high_correlation": "shows very high correlation with the target (r={corr})",
        "leak_reason_moderate_correlation": "shows moderately-high correlation with the target (r={corr})",
        "leak_reason_categorical_purity": "category '{value}' almost perfectly points to a single class",
        "leak_action_remove_column": "remove this column from the feature set",
        "leak_action_remove_or_check": "should very likely be removed — check what this column means",
        "leak_action_review_column": "review what this column represents, remove if needed",

        "t3_subheader": "⚙️ Feature Engineering and Preprocessing Pipeline",
        "t3_intro": "An automatic, memory-friendly preprocessing pipeline is configured for the different column types in your dataset.",
        "t3_scaler": "Numeric Feature Scaler:",
        "t3_max_cats": "Max Category Count (One-Hot Cap):",
        "t3_max_cats_help": "Prevents memory blowups by grouping rare categories under 'Other'.",
        "t3_tfidf": "Max TF-IDF Text Features:",
        "t3_test_split": "Test Data Split Ratio (%):",
        "t3_steps_header": "📋 Preprocessing Steps Summary:",
        "t3_image_header": "🖼️ Product Photo (Optional)",
        "t3_image_checkbox": "Extract visual features from product photos (ResNet50 + PCA)",
        "t3_image_help": "Photos are downloaded and processed on first run (can be slow); later runs are faster thanks to the on-disk cache. Requires an extra library.",
        "t3_image_no_col": "⚠️ No column that looks like a photo URL was found in the dataset; this option will be skipped.",
        "t3_image_col_select": "Which column contains the product photo URL?",
        "t3_image_pca_dim": "Image PCA Dimensions:",
        "t3_image_sample_n": "Number of Photos Sampled for PCA:",
        "t3_image_sample_help": "Only used to fit the PCA basis — during training/prediction, every row's actual photo is still processed.",
        "t3_image_note": "ℹ️ This column will now be included as a model feature. During single prediction, 'What-If' simulation and batch scoring, each row's real photo is downloaded and processed as it happens.",

        "t4_subheader": "🚀 AutoML Multi-Model Training & Benchmark",
        "t4_intro": "Algorithms are trained on the same train/test split, their metrics are compared, and the best-performing model is picked automatically.",
        "t4_model_select": "Select Models to Train:",
        "t4_sample_size_label": "⚡ Large Dataset Training Sample Size:",
        "t4_sample_size_help": "The current dataset has {n:,} rows. Pick a balanced sample size for faster results and memory safety.",
        "t4_dataset_size_info": "ℹ️ Dataset size: {n:,} rows (will be trained on the full dataset).",
        "t4_new_product_checkbox": "🆕 This model will be used for products not yet launched / with no sales",
        "t4_new_product_help": "When checked, columns reflecting a product's CURRENT traction (e.g. 'reviews', 'rating', 'sales') are entirely excluded from training.",
        "t4_new_product_excluded": "🆕 New-product mode: columns {cols} were excluded from training — because they are existing-traction signals.",
        "t4_new_product_none_found": "🆕 New-product mode is on, but no column resembling a traction signal was found in the dataset.",
        "t4_select_warning": "⚠️ Please select at least one model to train.",
        "t4_train_btn": "🚀 Start AutoML Model Training",
        "t4_training_spinner": "Training models, computing cross-validation and metrics…",
        "t4_train_success": "🎉 Training complete! Best-performing model: **{name}**",
        "t4_train_image_error": "⚠️ {error}",
        "t4_train_value_error": "⚠️ Training could not be completed: {error}",
        "t4_leakage_train_warning": "🚨 Warning: some of the selected features carry a high leakage risk. They may make the trained model look far more 'successful' than it would be in the real world — see the 'Data Leakage Check' section on tab 2.",
        "t4_leaderboard_header": "🏆 Model Leaderboard",
        "t4_best_model_header": "🌟 Best-Performing Model: `{name}`",
        "t4_fi_missing": "Feature importance could not be extracted directly for this model type.",

        "t5_subheader": "🔮 Live Product Success Prediction & 'What-If' Simulation",
        "t5_need_training": "⚠️ Please train a model first, on the **4. AutoML Model Training** tab!",
        "t5_subtab_single": "🧪 Single Product Prediction & Simulation",
        "t5_subtab_batch": "📁 Batch File Prediction (CSV)",
        "t5_form_intro": "Fill in the form below to instantly compute a new product's predicted success rate.",
        "t5_traction_note": "ℹ️ Fields like {cols} reflect the product's **current** performance. If you're evaluating a not-yet-launched product, enter 0 for these, or retrain the model with the '🆕 New Product' option on the **4. AutoML** tab.",
        "t5_random_fill_btn": "🎲 Fill With a Random Sample Product",
        "t5_photo_header": "🖼️ Product Photo",
        "t5_photo_upload_label": "Drag & drop or choose a photo",
        "t5_photo_upload_help": "This is the easiest way — JPG/PNG/WEBP. The photo is genuinely processed at prediction time.",
        "t5_photo_url_expander": "🔗 Use a photo link (URL) instead",
        "t5_photo_url_label": "Photo URL:",
        "t5_photo_preview_caption": "Uploaded photo",
        "t5_photo_none_caption": "No photo added yet",
        "t5_predict_btn": "🔮 Predict Product Success",
        "t5_result_header": "🎯 Prediction Result Card",
        "t5_gauge_title_pct": "Success Probability",
        "t5_gauge_title_score": "Predicted Score",
        "t5_level_high": "🟢 HIGH SUCCESS POTENTIAL",
        "t5_level_mid": "🟡 MODERATE — OPTIMIZATION RECOMMENDED",
        "t5_level_low": "🔴 LOW SUCCESS RISK / URGENT OPTIMIZATION",
        "t5_level_neutral": "📊 REGRESSION PREDICTION",
        "t5_prob_label": "Success Probability",
        "t5_prediction_label": "**Prediction:** {value}",
        "t5_model_label": "**Model:** `{value}`",
        "t5_proba_note": "ℹ️ This model does not support predict_proba; the value was derived from the predicted class.",
        "t5_drivers_header": "**⚖️ Most Influential Factors:**",
        "t5_success_label": "✅ Successful (1)",
        "t5_failure_label": "❌ Unsuccessful (0)",
        "t5_whatif_header": "🎛️ 'What-If' Sensitivity & Parameter Simulator",
        "t5_whatif_intro": "Change a chosen feature to simulate how the product's success probability reacts.",
        "t5_whatif_var_select": "Choose a Variable:",
        "t5_whatif_run_btn": "⚡ Run Simulation",
        "t5_batch_header": "📁 Batch Product Scoring",
        "t5_batch_image_note": "🖼️ Visual features are enabled: each row's photo will actually be downloaded and processed, so scoring large files with photos can be noticeably slower.",
        "t5_batch_upload_label": "Upload the CSV / Excel / JSON / ZIP file to score:",
        "t5_batch_score_btn": "🚀 Score All Products",
        "t5_batch_spinner": "Computing batch predictions…",
        "t5_batch_success": "{n:,} products scored successfully!",
        "t5_batch_download_btn": "📥 Download Scored Results (CSV)",

        "t6_subheader": "🤖 LLM Product Success Advisor & Strategic Report",
        "t6_live_active": "🟢 **{provider}** connected — live LLM analysis active.",
        "t6_heuristic_expander": "🔵 Built-in AI Advisor active — click to switch to a live LLM",
        "t6_heuristic_body": "Choose an **AI Provider** on the sidebar and enter your API key to activate a live LLM model (Gemini, Claude, DeepSeek or Qwen).\n\nWith a live LLM enabled:\n- ✅ 90-day growth roadmap in table format\n- ✅ Personalized SWOT specific to the product's features\n- ✅ Quick Wins (3 concrete actions for this week)\n- ✅ Concrete pricing & title optimization examples",
        "t6_product_summary_header": "📦 Predicted Product",
        "t6_product_summary_line": "Model: `{model}` &nbsp;|&nbsp; Prediction: `{pred}` &nbsp;|&nbsp; {prob}",
        "t6_analyze_btn": "🤖 Generate AI Analysis & Strategic Advice",
        "t6_analyze_spinner": "🧠 AI is analyzing — preparing strategic advice and SWOT…",
        "t6_need_prediction": "💡 Go to the **5. Live Prediction** tab and click 'Predict Product Success' for a product to trigger the AI analysis.",
        "t6_engine_label": "**Engine:** `{source}`",
        "t6_reanalyze_btn": "🔄 Re-analyze",
        "t6_reset_btn": "🧹 Reset",
        "t6_summary_quote": "> **Product Prediction:** {pred} &nbsp;|&nbsp; {prob} **Model:** `{model}`",
        "t6_schematic_header": "🗺️ Schematic View of Recommendations",
        "t6_schematic_caption": "Shows the flow from the strongest driving factors, to the prediction, to the recommended action categories.",
        "t6_roadmap_header": "🚀 90-Day Roadmap (Visual Timeline)",
        "t6_roadmap_phase1": "Launch & Validation",
        "t6_roadmap_phase2": "Scaling",
        "t6_roadmap_phase3": "Sustainability",
        "t6_export_header": "📥 Export Report",
        "t6_export_md_btn": "📄 Markdown Report (.md)",
        "t6_export_pdf_btn": "📑 PDF Report (.pdf)",

        "sankey_node_prediction_success": "High Success",
        "sankey_node_prediction_risk": "Needs Optimization",
        "sankey_cat_pricing": "Pricing",
        "sankey_cat_content": "Title / Content",
        "sankey_cat_marketing": "Marketing",
        "sankey_cat_trust": "Trust / Social Proof",
    },

    "de": {
        "app_title": "🚀 AI Product Success Predictor & Intelligence Studio",
        "app_subtitle": "Laden Sie einen beliebigen Datensatz hoch, trainieren Sie mit AutoML, erhalten Sie sofort die Erfolgswahrscheinlichkeit und KI-gestützte strategische Produktberatung.",
        "footer_badge": "📌 AI Product Predictor v2.0",
        "footer_caption": "AutoML + Sklearn + XGBoost + LightGBM + Mehrere LLM-Anbieter",
        "language_label": "🌐 Sprache",
        "theme_label": "🎨 Darstellung",
        "theme_dark": "🌙 Dunkel",
        "theme_light": "☀️ Hell",
        "logout_btn": "🚪 Abmelden",
        "logged_in_as": "Angemeldet als: **{name}**",

        "auth_app_tagline": "Melden Sie sich an, bevor Sie den Produkterfolg vorhersagen.",
        "auth_login_tab": "Anmelden",
        "auth_register_tab": "Konto erstellen",
        "auth_name_label": "Vollständiger Name",
        "auth_email_label": "E-Mail",
        "auth_password_label": "Passwort",
        "auth_password_confirm_label": "Passwort (Wiederholen)",
        "auth_login_btn": "🔓 Anmelden",
        "auth_register_btn": "✨ Konto erstellen",
        "auth_login_success": "Anmeldung erfolgreich! Sie werden weitergeleitet…",
        "auth_login_fail": "❌ E-Mail oder Passwort ist falsch.",
        "auth_register_success": "🎉 Konto erstellt! Sie können sich jetzt anmelden.",
        "auth_register_exists": "⚠️ Diese E-Mail-Adresse ist bereits registriert.",
        "auth_register_mismatch": "⚠️ Die Passwörter stimmen nicht überein.",
        "auth_register_weak": "⚠️ Das Passwort muss mindestens 6 Zeichen haben.",
        "auth_register_invalid_email": "⚠️ Bitte geben Sie eine gültige E-Mail-Adresse ein.",
        "auth_required_fields": "⚠️ Bitte füllen Sie alle Felder aus.",
        "auth_demo_hint": "Zum ersten Mal hier? Erstellen Sie über den Tab '{tab}' ein kostenloses Konto. Die Daten werden nur lokal auf diesem Rechner, gehasht, gespeichert.",

        "guide_expander_title": "👋 Neu hier? Lernen Sie in 1 Minute, wie diese App funktioniert",
        "guide_intro": "Dieses Tool funktioniert in 6 Schritten — folgen Sie ihnen einfach der Reihe nach:",
        "guide_step1": "**1. Datensatz & Ziel** — Wählen Sie einen integrierten Branchendatensatz oder laden Sie Ihren eigenen hoch (CSV/Excel/JSON/ZIP) und wählen Sie die Spalte, die Sie vorhersagen möchten (z. B. „erfolgreich?“).",
        "guide_step2": "**2. Automatische Analyse (EDA)** — Prüfen Sie Datenqualität, fehlende Werte und welche Merkmale mit Erfolg korrelieren, anhand automatischer Diagramme.",
        "guide_step3": "**3. Vorverarbeitung** — Die Standardeinstellungen funktionieren für die meisten Datensätze; passen Sie sie bei Bedarf an. Aktivieren Sie hier eine Produktfoto-Spalte, falls vorhanden.",
        "guide_step4": "**4. Modelltraining** — Trainieren und vergleichen Sie mit einem Klick mehrere Machine-Learning-Modelle; das beste wird automatisch ausgewählt.",
        "guide_step5": "**5. Live-Vorhersage & What-If** — Geben Sie die Daten eines neuen Produkts ein, sehen Sie sofort die Erfolgswahrscheinlichkeit und testen Sie „Was-wäre-wenn“-Szenarien.",
        "guide_step6": "**6. KI-Berater & Bericht** — Erhalten Sie eine personalisierte SWOT-Analyse und Wachstumsempfehlungen von der KI und laden Sie sie als PDF-/Markdown-Bericht herunter.",
        "guide_tip": "💡 Tipp: Sprache, Darstellung (hell/dunkel) und KI-Anbieter können Sie jederzeit über die Seitenleiste ändern.",

        "sidebar_data_control_header": "⚙️ Daten- & Modellsteuerung",
        "dataset_source_label": "Datenquelle:",
        "dataset_source_builtin": "Integrierte Branchendatensätze",
        "dataset_source_custom": "Eigenen Datensatz hochladen (CSV/Excel/JSON)",
        "sector_select_label": "Branche auswählen:",
        "load_dataset_btn": "🔄 Datensatz laden",
        "dataset_loaded_msg": "{name} geladen!",
        "upload_file_label": "Datendatei hochladen:",
        "upload_file_help": "Per Drag & Drop ablegen oder Datei auswählen (ZIP-Archive werden unterstützt).",
        "zip_no_valid_file": "⚠️ Im hochgeladenen ZIP-Archiv wurde keine gültige Datendatei (CSV/Excel/JSON) gefunden.",
        "zip_single_info": "📦 ZIP-Archiv: `{filename}` ({size})",
        "zip_multi_found": "📦 **{n} Datendateien im ZIP gefunden:**",
        "zip_choose_file": "Zu ladende Datei auswählen:",
        "zip_read_error": "Fehler beim Lesen des ZIP-Archivs: {error}",
        "process_upload_btn": "📥 Hochgeladene Daten verarbeiten",
        "upload_success": "'{name}' erfolgreich geladen! ({n:,} Zeilen)",
        "upload_error": "Fehler beim Laden der Daten: {error}",

        "llm_settings_header": "🤖 KI (LLM) Einstellungen",
        "llm_provider_label": "KI-Anbieter wählen:",
        "llm_api_key_label": "{provider} API-Schlüssel (optional):",
        "llm_api_key_help": "Wenn ein Schlüssel eingegeben wird, wird das Live-{provider}-Modell verwendet. Andernfalls übernimmt der intelligente integrierte KI-Berater.",
        "llm_active_live": "🟢 Live-{provider}-API aktiv.",
        "llm_active_heuristic": "🔵 Intelligenter integrierter Heuristik-KI-Motor aktiv.",
        "llm_model_override_label": "Modellname (optional, leer lassen für Standard):",
        "llm_model_override_expander_title": "⚙️ Erweiterte Einstellung: bestimmte Modellversion verwenden (optional)",
        "llm_model_override_help": (
            "Dieses Feld legt fest, **welche genaue Modellversion** des gewählten Anbieters "
            "aufgerufen wird — die meisten Nutzer müssen es nie anfassen. Bleibt es leer, wird "
            "automatisch das empfohlene Standardmodell des Anbieters (**{default_model}**) verwendet. "
            "Füllen Sie es nur aus, wenn Sie gezielt eine andere/neuere Modellversion testen möchten — "
            "tragen Sie den genauen Modellnamen aus der Dokumentation des Anbieters ein (z. B. {example})."
        ),

        "active_extended_lang_note": "🌍 Aktive Sprache: **{lang}** (KI-generiert)",
        "extended_lang_header": "Weitere Sprachen (KI-gestützt)",
        "extended_lang_help": (
            "Über Türkisch/Englisch/Deutsch hinaus können Sie aus **Dutzenden Sprachen weltweit** "
            "wählen — die Oberflächentexte werden einmalig mit dem oben eingerichteten KI-Anbieter "
            "übersetzt und dann zwischengespeichert; jede weitere Nutzung ist sofort verfügbar. "
            "Deshalb ist ein API-Schlüssel erforderlich."
        ),
        "extended_lang_select_label": "Sprache wählen:",
        "extended_lang_apply_btn": "🌍 Zu dieser Sprache wechseln",
        "extended_lang_need_key": "Um diese Sprache zu nutzen, müssen Sie oben zuerst einen {provider}-API-Schlüssel eingeben — ohne Schlüssel sind nur Türkisch/Englisch/Deutsch verfügbar.",
        "extended_lang_translating": "Wird ins {lang} übersetzt (nur beim ersten Mal)...",
        "extended_lang_success": "✅ Die Oberfläche ist jetzt auf {lang}!",
        "extended_lang_fail": "❌ Übersetzung fehlgeschlagen — überprüfen Sie Ihren API-Schlüssel und versuchen Sie es erneut.",
        "extended_lang_advisor_note": "ℹ️ Der KI-Berater (Tab 6) schreibt direkt in dieser Sprache, sobald ein API-Schlüssel eingegeben ist. Ohne Schlüssel unterstützt der integrierte Heuristik-Berater nur Türkisch/Englisch/Deutsch und weicht dann auf Türkisch aus.",

        "accent_label": "🎨 Farbpalette",
        "accent_auto": "🎲 Automatisch (erneuert sich bei jeder Anmeldung)",
        "accent_indigo": "Indigo & Lila",
        "accent_emerald": "Smaragd & Türkis",
        "accent_ocean": "Ozeanblau",
        "accent_sunset": "Sonnenuntergang",
        "accent_gold": "Gold & Bernstein",
        "accent_berry": "Beere (Lila & Pink)",
        "accent_help": "Wählen Sie selbst eine Farbe, oder belassen Sie es bei 'Automatisch', damit sich die App bei jeder Anmeldung mit einer neuen Farbe erneuert. Ihre Wahl wird in Ihrem Konto gespeichert.",
        "accent_shuffle_btn": "🔄 Jetzt erneuern",
        "accent_saved_note": "💾 Ihre Einstellung wurde in Ihrem Konto gespeichert.",

        "tab1_title": "📁 1. Datensatz & Zielauswahl",
        "tab2_title": "📊 2. Automatisierte explorative Analyse (EDA)",
        "tab3_title": "⚙️ 3. Vorverarbeitung & Pipeline",
        "tab4_title": "🚀 4. AutoML-Modelltraining",
        "tab5_title": "🔮 5. Live-Vorhersage & Simulator",
        "tab6_title": "🤖 6. LLM-Produktberater & Bericht",

        "t1_subheader": "📁 Datensatzprüfung und Auswahl der Zielvariable",
        "t1_active_dataset": "Aktiver Datensatz: **{name}** — **{rows:,} Zeilen**, **{cols} Spalten**",
        "t1_target_select": "🎯 Vorherzusagende Zielspalte:",
        "t1_target_select_help": "Diese Liste zeigt die Spaltennamen aus Ihrer hochgeladenen Datei — zur besseren Lesbarkeit mit Leerzeichen und Großbuchstaben formatiert. Da die Namen direkt aus Ihrer eigenen Datei stammen, können wir sie nicht in eine andere Sprache übersetzen; wenn Sie andere Spaltennamen möchten, benennen Sie die Kopfzeilen in Ihrer Quelldatei um und laden Sie sie erneut hoch.",
        "t1_derive_expander": "🛠️ Benutzerdefiniertes Erfolgsziel aus einer Regel ableiten (optional)",
        "t1_derive_help": "Erstellen Sie aus einer vorhandenen numerischen Spalte ein neues Erfolgs-/Misserfolgsziel (z. B. Umsatz > 500 oder Bewertung >= 4,5).",
        "t1_derive_base_col": "Basierend auf welcher Spalte?",
        "t1_derive_condition": "Bedingung:",
        "t1_derive_threshold_pct": "Schwellenwert (oberes Perzentil %):",
        "t1_derive_threshold": "Schwellenwert:",
        "t1_derive_new_name": "Name der neuen Zielspalte:",
        "t1_derive_btn": "✨ Benutzerdefiniertes Ziel erstellen und auswählen",
        "t1_derive_success": "Zielspalte '{name}' erfolgreich erstellt und ausgewählt!",
        "t1_derive_leak_warning": "⚠️ Um Data Leakage zu vermeiden, wurde die Spalte '{col}' automatisch aus dem Feature-Set entfernt — da das Ziel direkt daraus abgeleitet wird.",
        "t1_task_type_line": "**Aufgabentyp:** `{task}` | **Untertyp:** `{subtype}`",
        "t1_task_classification": "Klassifikation",
        "t1_task_regression": "Regression",
        "t1_preview_header": "🔍 Datenvorschau (erste 50 Zeilen)",
        "t1_column_types_header": "🧩 Automatisch erkannte Spaltentypen",
        "t1_numeric": "🔢 Numerisch ({n})",
        "t1_categorical": "🏷️ Kategorisch ({n})",
        "t1_text": "📝 Text ({n})",
        "t1_boolean": "🔘 Boolesch ({n})",
        "t1_none": "Keine",

        "t2_subheader": "📊 Automatisierte explorative Datenanalyse (Auto-EDA)",
        "t2_kpi_rows": "Zeilen insgesamt",
        "t2_kpi_cols": "Merkmale insgesamt",
        "t2_kpi_missing": "Fehlende Werte",
        "t2_kpi_duplicates": "Doppelte Zeilen",
        "t2_insights_header": "💡 Statistische Kurzergebnisse",
        "t2_feature_deep_dive": "🔬 Vertiefung: Zusammenhang zwischen Merkmal und Ziel",
        "t2_feature_select": "Zu untersuchendes Merkmal:",
        "t2_corr_insufficient": "Nicht genügend numerische Merkmale für eine Korrelations-Heatmap.",
        "t2_leakage_header": "🚨 Data-Leakage-Prüfung",
        "t2_leakage_none": "✅ Kein offensichtliches Data-Leakage-Signal erkannt.",
        "t2_leakage_found": "⚠️ Die folgenden Spalten zeigen eine verdächtig starke Beziehung zum Ziel — sie könnten dazu führen, dass das Modell 'zu leicht' lernt, um in der Praxis zu funktionieren:",
        "t2_leakage_row": "**`{col}`** — {reason} *(empfohlene Maßnahme: {action})*",
        "t2_gap_header": "🕳️ Datenlücken & Qualitätshinweise",
        "t2_gap_high_missing": "**`{col}`** fehlt zu **{pct:.1f}%** — das ist ein hoher Anteil; überprüfen Sie Ihre Imputationsstrategie, bevor Sie es ins Modell aufnehmen.",
        "t2_gap_low_variance": "**`{col}`** zeigt kaum Variation (ein Wert dominiert) — es liefert dem Modell möglicherweise keine unterscheidende Information.",
        "t2_gap_none": "✅ Keine nennenswerten Datenlücken oder Qualitätsprobleme gefunden.",
        "leak_reason_duplicate_of_target": "hat exakt dieselben Werte wie die Zielspalte",
        "leak_reason_high_correlation": "zeigt eine sehr hohe Korrelation mit dem Ziel (r={corr})",
        "leak_reason_moderate_correlation": "zeigt eine mittel-hohe Korrelation mit dem Ziel (r={corr})",
        "leak_reason_categorical_purity": "Kategorie '{value}' verweist fast ausschließlich auf eine einzige Klasse",
        "leak_action_remove_column": "entfernen Sie diese Spalte aus dem Feature-Set",
        "leak_action_remove_or_check": "sollte höchstwahrscheinlich entfernt werden — prüfen Sie die Bedeutung der Spalte",
        "leak_action_review_column": "prüfen Sie die Bedeutung der Spalte, entfernen Sie sie bei Bedarf",

        "t3_subheader": "⚙️ Feature Engineering und Vorverarbeitungs-Pipeline",
        "t3_intro": "Für die verschiedenen Spaltentypen in Ihrem Datensatz wird eine automatische, speicherschonende Vorverarbeitungs-Pipeline konfiguriert.",
        "t3_scaler": "Skalierer für numerische Merkmale:",
        "t3_max_cats": "Maximale Kategorienzahl (One-Hot-Obergrenze):",
        "t3_max_cats_help": "Verhindert Speicherüberläufe, indem seltene Kategorien unter 'Sonstige' zusammengefasst werden.",
        "t3_tfidf": "Maximale TF-IDF-Textmerkmale:",
        "t3_test_split": "Anteil der Testdaten (%):",
        "t3_steps_header": "📋 Übersicht der Vorverarbeitungsschritte:",
        "t3_image_header": "🖼️ Produktfoto (optional)",
        "t3_image_checkbox": "Visuelle Merkmale aus Produktfotos extrahieren (ResNet50 + PCA)",
        "t3_image_help": "Fotos werden beim ersten Lauf heruntergeladen und verarbeitet (kann langsam sein); spätere Läufe sind dank des Festplatten-Caches schneller. Erfordert eine zusätzliche Bibliothek.",
        "t3_image_no_col": "⚠️ Im Datensatz wurde keine Spalte gefunden, die wie eine Foto-URL aussieht; diese Option wird übersprungen.",
        "t3_image_col_select": "Welche Spalte enthält die URL des Produktfotos?",
        "t3_image_pca_dim": "PCA-Dimensionen für Bilder:",
        "t3_image_sample_n": "Anzahl der für PCA gesampelten Fotos:",
        "t3_image_sample_help": "Wird nur verwendet, um die PCA-Basis zu bilden — beim Training/der Vorhersage wird trotzdem das tatsächliche Foto jeder Zeile verarbeitet.",
        "t3_image_note": "ℹ️ Diese Spalte wird nun als Modellmerkmal einbezogen. Bei Einzelvorhersage, 'What-If'-Simulation und Batch-Bewertung wird jeweils das tatsächliche Foto jeder Zeile in Echtzeit heruntergeladen und verarbeitet.",

        "t4_subheader": "🚀 AutoML Multi-Modell-Training & Vergleich",
        "t4_intro": "Die Algorithmen werden auf demselben Trainings-/Testsplit trainiert, ihre Metriken werden verglichen, und das leistungsstärkste Modell wird automatisch ausgewählt.",
        "t4_model_select": "Zu trainierende Modelle auswählen:",
        "t4_sample_size_label": "⚡ Stichprobengröße für großes Trainings-Dataset:",
        "t4_sample_size_help": "Der aktuelle Datensatz hat {n:,} Zeilen. Wählen Sie eine ausgewogene Stichprobengröße für schnellere Ergebnisse und Speichersicherheit.",
        "t4_dataset_size_info": "ℹ️ Datensatzgröße: {n:,} Zeilen (wird mit dem vollständigen Datensatz trainiert).",
        "t4_new_product_checkbox": "🆕 Dieses Modell wird für noch nicht eingeführte Produkte ohne Verkäufe verwendet",
        "t4_new_product_help": "Wenn aktiviert, werden Spalten, die die AKTUELLE Zugkraft eines Produkts widerspiegeln (z. B. 'reviews', 'rating', 'sales'), vollständig vom Training ausgeschlossen.",
        "t4_new_product_excluded": "🆕 Neuprodukt-Modus: Die Spalten {cols} wurden vom Training ausgeschlossen — da es sich um bestehende Zugkraft-Signale handelt.",
        "t4_new_product_none_found": "🆕 Der Neuprodukt-Modus ist aktiv, aber im Datensatz wurde keine Spalte gefunden, die einem Zugkraft-Signal ähnelt.",
        "t4_select_warning": "⚠️ Bitte wählen Sie mindestens ein Modell zum Trainieren aus.",
        "t4_train_btn": "🚀 AutoML-Modelltraining starten",
        "t4_training_spinner": "Modelle werden trainiert, Kreuzvalidierung und Metriken werden berechnet…",
        "t4_train_success": "🎉 Training abgeschlossen! Bestes Modell: **{name}**",
        "t4_train_image_error": "⚠️ {error}",
        "t4_train_value_error": "⚠️ Training konnte nicht abgeschlossen werden: {error}",
        "t4_leakage_train_warning": "🚨 Warnung: Einige der ausgewählten Merkmale bergen ein hohes Leakage-Risiko. Dadurch könnte das trainierte Modell in der Praxis deutlich weniger 'erfolgreich' sein, als es hier erscheint — siehe Abschnitt 'Data-Leakage-Prüfung' auf Tab 2.",
        "t4_leaderboard_header": "🏆 Modell-Bestenliste",
        "t4_best_model_header": "🌟 Bestes Modell: `{name}`",
        "t4_fi_missing": "Für diesen Modelltyp konnte die Feature-Wichtigkeit nicht direkt extrahiert werden.",

        "t5_subheader": "🔮 Live-Produkterfolgsvorhersage & 'What-If'-Simulation",
        "t5_need_training": "⚠️ Bitte trainieren Sie zuerst ein Modell auf dem Tab **4. AutoML-Modelltraining**!",
        "t5_subtab_single": "🧪 Einzelprodukt-Vorhersage & Simulation",
        "t5_subtab_batch": "📁 Batch-Dateivorhersage (CSV)",
        "t5_form_intro": "Füllen Sie das folgende Formular aus, um die vorhergesagte Erfolgsquote eines neuen Produkts sofort zu berechnen.",
        "t5_traction_note": "ℹ️ Felder wie {cols} spiegeln die **aktuelle** Leistung des Produkts wider. Wenn Sie ein noch nicht eingeführtes Produkt bewerten, geben Sie hier 0 ein oder trainieren Sie das Modell mit der Option '🆕 Neues Produkt' auf dem Tab **4. AutoML** neu.",
        "t5_random_fill_btn": "🎲 Mit einem zufälligen Beispielprodukt füllen",
        "t5_photo_header": "🖼️ Produktfoto",
        "t5_photo_upload_label": "Foto per Drag & Drop ablegen oder auswählen",
        "t5_photo_upload_help": "Das ist der einfachste Weg — JPG/PNG/WEBP. Das Foto wird zum Vorhersagezeitpunkt tatsächlich verarbeitet.",
        "t5_photo_url_expander": "🔗 Stattdessen einen Foto-Link (URL) verwenden",
        "t5_photo_url_label": "Foto-URL:",
        "t5_photo_preview_caption": "Hochgeladenes Foto",
        "t5_photo_none_caption": "Noch kein Foto hinzugefügt",
        "t5_predict_btn": "🔮 Produkterfolg vorhersagen",
        "t5_result_header": "🎯 Ergebniskarte der Vorhersage",
        "t5_gauge_title_pct": "Erfolgswahrscheinlichkeit",
        "t5_gauge_title_score": "Vorhergesagter Wert",
        "t5_level_high": "🟢 HOHES ERFOLGSPOTENZIAL",
        "t5_level_mid": "🟡 MÄSSIG — OPTIMIERUNG EMPFOHLEN",
        "t5_level_low": "🔴 GERINGES ERFOLGSPOTENZIAL / DRINGENDE OPTIMIERUNG",
        "t5_level_neutral": "📊 REGRESSIONSVORHERSAGE",
        "t5_prob_label": "Erfolgswahrscheinlichkeit",
        "t5_prediction_label": "**Vorhersage:** {value}",
        "t5_model_label": "**Modell:** `{value}`",
        "t5_proba_note": "ℹ️ Dieses Modell unterstützt predict_proba nicht; der Wert wurde aus der vorhergesagten Klasse abgeleitet.",
        "t5_drivers_header": "**⚖️ Einflussreichste Faktoren:**",
        "t5_success_label": "✅ Erfolgreich (1)",
        "t5_failure_label": "❌ Nicht erfolgreich (0)",
        "t5_whatif_header": "🎛️ 'What-If'-Sensitivitäts- und Parametersimulator",
        "t5_whatif_intro": "Ändern Sie ein ausgewähltes Merkmal, um zu simulieren, wie die Erfolgswahrscheinlichkeit des Produkts reagiert.",
        "t5_whatif_var_select": "Variable auswählen:",
        "t5_whatif_run_btn": "⚡ Simulation ausführen",
        "t5_batch_header": "📁 Batch-Produktbewertung",
        "t5_batch_image_note": "🖼️ Visuelle Merkmale sind aktiviert: Das Foto jeder Zeile wird tatsächlich heruntergeladen und verarbeitet, daher kann die Bewertung großer Dateien mit Fotos deutlich langsamer sein.",
        "t5_batch_upload_label": "CSV-/Excel-/JSON-/ZIP-Datei zur Bewertung hochladen:",
        "t5_batch_score_btn": "🚀 Alle Produkte bewerten",
        "t5_batch_spinner": "Batch-Vorhersagen werden berechnet…",
        "t5_batch_success": "{n:,} Produkte erfolgreich bewertet!",
        "t5_batch_download_btn": "📥 Bewertete Ergebnisse herunterladen (CSV)",

        "t6_subheader": "🤖 LLM-Produkterfolgsberater & Strategiebericht",
        "t6_live_active": "🟢 **{provider}** verbunden — Live-LLM-Analyse aktiv.",
        "t6_heuristic_expander": "🔵 Integrierter KI-Berater aktiv — Klicken, um zu einem Live-LLM zu wechseln",
        "t6_heuristic_body": "Wählen Sie in der Seitenleiste einen **KI-Anbieter** und geben Sie Ihren API-Schlüssel ein, um ein Live-LLM-Modell zu aktivieren (Gemini, Claude, DeepSeek oder Qwen).\n\nMit aktiviertem Live-LLM:\n- ✅ 90-Tage-Wachstumsfahrplan im Tabellenformat\n- ✅ Personalisierte SWOT-Analyse passend zu den Produktmerkmalen\n- ✅ Quick Wins (3 konkrete Maßnahmen für diese Woche)\n- ✅ Konkrete Beispiele zur Preis- und Titeloptimierung",
        "t6_product_summary_header": "📦 Vorhergesagtes Produkt",
        "t6_product_summary_line": "Modell: `{model}` &nbsp;|&nbsp; Vorhersage: `{pred}` &nbsp;|&nbsp; {prob}",
        "t6_analyze_btn": "🤖 KI-Analyse & Strategieempfehlungen erstellen",
        "t6_analyze_spinner": "🧠 Die KI analysiert — Strategieempfehlungen und SWOT werden vorbereitet…",
        "t6_need_prediction": "💡 Gehen Sie zum Tab **5. Live-Vorhersage** und klicken Sie für ein Produkt auf 'Produkterfolg vorhersagen', um die KI-Analyse auszulösen.",
        "t6_engine_label": "**Engine:** `{source}`",
        "t6_reanalyze_btn": "🔄 Erneut analysieren",
        "t6_reset_btn": "🧹 Zurücksetzen",
        "t6_summary_quote": "> **Produktvorhersage:** {pred} &nbsp;|&nbsp; {prob} **Modell:** `{model}`",
        "t6_schematic_header": "🗺️ Schematische Ansicht der Empfehlungen",
        "t6_schematic_caption": "Zeigt den Fluss von den stärksten Einflussfaktoren über die Vorhersage bis zu den empfohlenen Aktionskategorien.",
        "t6_roadmap_header": "🚀 90-Tage-Fahrplan (visuelle Zeitleiste)",
        "t6_roadmap_phase1": "Markteinführung & Validierung",
        "t6_roadmap_phase2": "Skalierung",
        "t6_roadmap_phase3": "Nachhaltigkeit",
        "t6_export_header": "📥 Bericht exportieren",
        "t6_export_md_btn": "📄 Markdown-Bericht (.md)",
        "t6_export_pdf_btn": "📑 PDF-Bericht (.pdf)",

        "sankey_node_prediction_success": "Hoher Erfolg",
        "sankey_node_prediction_risk": "Optimierung nötig",
        "sankey_cat_pricing": "Preisgestaltung",
        "sankey_cat_content": "Titel / Inhalt",
        "sankey_cat_marketing": "Marketing",
        "sankey_cat_trust": "Vertrauen / Social Proof",
    },
}


def get_lang() -> str:
    """Returns the currently active language code, defaulting to Turkish."""
    return st.session_state.get("lang", DEFAULT_LANG)


def set_lang(lang_code: str) -> None:
    """Activates `lang_code` — native (tr/en/de) or extended, as long as it's
    a language this app knows about (has a display name for)."""
    if lang_code in ALL_LANGUAGES:
        st.session_state["lang"] = lang_code


def is_natively_supported(lang_code: str) -> bool:
    """True for the 3 hand-translated languages that never need an API key
    (also the only languages the local heuristic AI advisor can write in)."""
    return lang_code in TRANSLATIONS


def language_name_en(code: str) -> str:
    """Plain English name for `code`, for use inside an LLM instruction
    ("write your answer in {this}") — never shown directly in the UI."""
    return LANGUAGE_NAMES_EN.get(code, code)


def t(key: str, **kwargs: Any) -> str:
    """
    Translates `key` into the active language, formatting with kwargs if given.
    For an extended (non-native) language, looks in the dynamically generated
    cache; any key missing there (translation still pending, or that batch
    failed) falls back to English, then Turkish, then the raw key — so a
    partially-translated language never crashes or shows a blank string.
    """
    lang = get_lang()
    if lang in TRANSLATIONS:
        text = TRANSLATIONS[lang].get(key)
    else:
        text = _dynamic_cache.get(lang, {}).get(key)
        if text is None:
            text = TRANSLATIONS["en"].get(key)
    if text is None:
        text = TRANSLATIONS.get(DEFAULT_LANG, {}).get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


# ---------------------------------------------------------------------------
# Dynamic (LLM-generated) translation for extended languages
# ---------------------------------------------------------------------------
def _cache_path(lang_code: str) -> str:
    return os.path.join(_I18N_CACHE_DIR, f"{lang_code}.json")


def _load_disk_cache(lang_code: str) -> Optional[Dict[str, str]]:
    path = _cache_path(lang_code)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_disk_cache(lang_code: str, data: Dict[str, str]) -> None:
    try:
        os.makedirs(_I18N_CACHE_DIR, exist_ok=True)
        with open(_cache_path(lang_code), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # best-effort disk cache; a miss just means re-translating next time


def has_dynamic_language(lang_code: str) -> bool:
    """True if `lang_code` is already translated and ready to use — either
    already loaded this run, or sitting in the on-disk cache from a previous
    run (by this user or, since the cache directory is shared, anyone else
    who already generated it) — with no LLM call needed right now."""
    if lang_code in TRANSLATIONS:
        return True
    if lang_code in _dynamic_cache:
        return True
    return _load_disk_cache(lang_code) is not None


def ensure_dynamic_language(
    lang_code: str,
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
) -> bool:
    """
    Makes sure a full UI translation exists for `lang_code` beyond the three
    natively hand-translated languages, generating it once via the
    configured LLM provider and caching it to disk (keyed only by language,
    so the very first person to request e.g. Spanish generates it and every
    user afterwards gets it instantly, cache hit, no new API call).

    Returns True once a usable translation is active (freshly generated OR
    already cached), False if it could not be generated at all (e.g. an
    invalid API key) — callers should keep the UI in its previous language
    rather than switching to a half-translated one.
    """
    if lang_code in TRANSLATIONS:
        return True
    if has_dynamic_language(lang_code):
        if lang_code not in _dynamic_cache:
            _dynamic_cache[lang_code] = _load_disk_cache(lang_code) or {}
        return True

    from src.llm.providers import call_llm, LLMCallError

    base = TRANSLATIONS["en"]
    items = list(base.items())
    batch_size = 60  # keeps each request small enough to translate reliably
    translated: Dict[str, str] = {}
    lang_name = language_name_en(lang_code)

    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i + batch_size])
        prompt = (
            f"Translate the values of this JSON object into {lang_name} ({lang_code}). "
            "This is user-interface copy for a web application. Rules:\n"
            "- Return ONLY a valid JSON object with exactly the same keys as the input.\n"
            "- Keep any {placeholder} tokens (e.g. {name}, {n}, {provider}) unchanged and in a "
            "grammatically natural position for the target language.\n"
            "- Keep a leading emoji in a value unchanged.\n"
            "- Keep Markdown formatting characters (**, `, #, -, |) unchanged.\n"
            "- Translate naturally and concisely for a software UI; do not add commentary.\n\n"
            f"{json.dumps(batch, ensure_ascii=False)}"
        )
        try:
            raw = call_llm(
                provider=provider,
                api_key=api_key,
                system_prompt="You are a precise JSON-only translation engine for software UI strings. You output nothing but the requested JSON object.",
                user_prompt=prompt,
                model_name=model_name,
                temperature=0.2,
            )
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                translated.update({k: v for k, v in parsed.items() if isinstance(v, str) and v.strip()})
        except (LLMCallError, json.JSONDecodeError, Exception):
            continue  # this batch's keys simply fall back to English in t()

    if not translated:
        return False

    # Any key a batch failed to translate just stays in English — t() would
    # already fall back there, but resolving it now means the cache file is
    # a complete, self-contained snapshot for next time.
    full = dict(base)
    full.update(translated)
    _dynamic_cache[lang_code] = full
    _save_disk_cache(lang_code, full)
    return True
