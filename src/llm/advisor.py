"""
LLM AI Product Advisor Module
Integrates multiple LLM providers (Google Gemini, Anthropic Claude, DeepSeek,
Qwen) with an intelligent local heuristic advisor fallback that always works
even without any API key configured.
"""

from typing import Dict, List, Optional, Any
import os
import numpy as np
from dotenv import load_dotenv

from src.llm.providers import call_llm, LLMCallError, get_provider_meta
from src.llm.prompts import get_system_prompt, get_product_analysis_prompt
from src.utils.i18n import language_name_en

load_dotenv()


class AIProductAdvisor:
    """Provides LLM-powered strategic product success insights & SWOT analysis."""

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None,
                 model_name: Optional[str] = None, lang: str = "tr"):
        self.provider = (provider or "gemini").lower()
        meta = get_provider_meta(self.provider)
        self.api_key = api_key or os.getenv(meta["env_key"])
        self.model_name = model_name
        # `self.lang` keeps the FULL requested language (which may be one of
        # the ~40 "extended" languages generated on demand for the UI, see
        # src/utils/i18n.py) — a live LLM call can write directly in any of
        # those via an explicit instruction below. The local heuristic
        # advisor has only three hand-written template sets, so it always
        # falls back to `self.heuristic_lang` instead.
        self.lang = lang or "tr"
        self.heuristic_lang = self.lang if self.lang in ("tr", "en", "de") else "tr"

    def analyze_product_success(
        self,
        product_dict: Dict[str, Any],
        prediction_label: str,
        probability: float,
        best_model_name: str,
        model_metric: str,
        top_drivers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates full strategic AI analysis for a predicted product.
        Uses the configured live LLM provider if an API key is present,
        otherwise falls back to the smart local rule-based heuristic advisor.
        """
        prob_pct = probability * 100 if probability <= 1.0 else probability

        # Format features string
        features_str = "\n".join([f"- **{k}**: `{v}`" for k, v in product_dict.items()])

        # Format drivers string — prefer the humanized "FeatureLabel" (e.g.
        # "Kategori: Elektronik") over the raw pipeline-internal "Feature"
        # name (e.g. "category_Elektronik") so the LLM's own writeup talks
        # about the product in plain language instead of encoded columns.
        if top_drivers:
            drivers_str = "\n".join([
                f"- **{d.get('FeatureLabel', d.get('Feature', 'Feature'))}**: Impact `{d.get('Impact', 0.0):.4f}`"
                for d in top_drivers[:5]
            ])
        else:
            drivers_str = "- Based on overall model correlations."

        # The hand-written prompt templates only exist in tr/en/de. For any
        # other (dynamically-added) UI language, build the prompt in English
        # and append an explicit instruction so the LIVE model still writes
        # its answer in the user's actual chosen language — this is what
        # lets "add a language" scale to anything the LLM itself supports,
        # without hand-writing a new prompt template per language.
        template_lang = self.lang if self.lang in ("tr", "en", "de") else "en"
        prompt = get_product_analysis_prompt(
            template_lang,
            product_features=features_str,
            prediction_label=prediction_label,
            probability_percent=prob_pct,
            best_model_name=best_model_name,
            model_metric=model_metric,
            top_drivers=drivers_str
        )
        system_prompt = get_system_prompt(template_lang)
        if self.lang not in ("tr", "en", "de"):
            target_name = language_name_en(self.lang)
            system_prompt += (
                f"\n\nIMPORTANT: Write your ENTIRE response in {target_name}, not English. "
                f"Keep the same Markdown structure, section headings, and tables — translate "
                f"their text into {target_name} as well."
            )

        provider_meta = get_provider_meta(self.provider)
        last_error = None

        if self.api_key:
            try:
                text = call_llm(
                    provider=self.provider,
                    api_key=self.api_key,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    model_name=self.model_name,
                )
                return {
                    "analysis_markdown": text,
                    "source": f"{provider_meta['label']} (Live LLM) ✨"
                }
            except LLMCallError as err:
                last_error = str(err)
                print(f"{provider_meta['label']} API call failed, switching to local advisor: {last_error}")

        # Local Intelligent Heuristic Advisor Fallback — only tr/en/de exist
        # as hand-written templates, so an extended UI language quietly
        # falls back to Turkish here (self.heuristic_lang handles that).
        source = _t(self.heuristic_lang, "heuristic_source")
        if last_error:
            # Surface *why* we fell back instead of silently pretending no
            # provider was configured — otherwise a bad/expired API key or a
            # deprecated model name looks identical to "no key entered" in the UI.
            source += f" — {_t(self.heuristic_lang, 'heuristic_fallback_reason')}: {last_error[:160]}"

        return {
            "analysis_markdown": self._generate_heuristic_analysis(
                product_dict=product_dict,
                prediction_label=prediction_label,
                probability_percent=prob_pct,
                best_model_name=best_model_name,
                top_drivers=top_drivers
            ),
            "source": source
        }

    def _generate_heuristic_analysis(
        self,
        product_dict: Dict[str, Any],
        prediction_label: str,
        probability_percent: float,
        best_model_name: str,
        top_drivers: List[Dict[str, Any]]
    ) -> str:
        """Constructs an exhaustive, high-quality strategic report deterministically, in the active language."""
        lang = self.heuristic_lang
        L = _HEURISTIC_STRINGS[lang]
        is_high_success = probability_percent >= 60.0
        status_emoji = "🟢" if is_high_success else ("🟡" if probability_percent >= 40.0 else "🔴")

        strengths, weaknesses, opportunities, threats = [], [], [], []

        for k, v in product_dict.items():
            k_low = k.lower()
            try:
                val_num = float(v)
                if "rating" in k_low or "puan" in k_low:
                    if val_num >= 4.3:
                        strengths.append(L["rating_high"].format(v=val_num))
                    elif val_num < 4.0:
                        weaknesses.append(L["rating_low"].format(v=val_num))
                elif "price" in k_low or "fiyat" in k_low or "goal" in k_low:
                    if val_num > 150:
                        threats.append(L["price_high"].format(v=val_num))
                    else:
                        strengths.append(L["price_low"].format(v=val_num))
                elif "discount" in k_low or "indirim" in k_low:
                    if val_num > 15:
                        strengths.append(L["discount_high"].format(v=val_num))
                    else:
                        opportunities.append(L["discount_opportunity"])
                elif "reviews" in k_low or "review_count" in k_low:
                    if val_num > 100:
                        strengths.append(L["reviews_high"].format(v=int(val_num)))
                    else:
                        weaknesses.append(L["reviews_low"].format(v=int(val_num)))
            except Exception:
                pass

            if isinstance(v, (bool, np.bool_)) or str(v).lower() in ["true", "false"]:
                val_bool = bool(v) if isinstance(v, (bool, np.bool_)) else str(v).lower() == "true"
                if "prime" in k_low or "fast_shipping" in k_low:
                    strengths.append(L["prime_yes"]) if val_bool else weaknesses.append(L["prime_no"])
                elif "video" in k_low:
                    strengths.append(L["video_yes"]) if val_bool else opportunities.append(L["video_opportunity"])
                elif "api" in k_low or "free_tier" in k_low:
                    if val_bool:
                        strengths.append(L["api_yes"])

        if not strengths:
            strengths.append(L["default_strength"])
        if not weaknesses:
            weaknesses.append(L["default_weakness"])
        if not opportunities:
            opportunities.append(L["default_opportunity"])
        if not threats:
            threats.append(L["default_threat"])

        outlook = L["outlook_high"] if is_high_success else L["outlook_low"]

        md = L["report_template"].format(
            status_emoji=status_emoji,
            probability_percent=probability_percent,
            prediction_label=prediction_label,
            outlook=outlook,
            strengths="\n".join([f"- {s}" for s in strengths]),
            weaknesses="\n".join([f"- {w}" for w in weaknesses]),
            opportunities="\n".join([f"- {o}" for o in opportunities]),
            threats="\n".join([f"- {t}" for t in threats]),
        )
        return md


def _t(lang: str, key: str) -> str:
    return _HEURISTIC_STRINGS.get(lang, _HEURISTIC_STRINGS["tr"]).get(key, key)


# ---------------------------------------------------------------------------
# Heuristic (no-API-key) advisor strings, per language.
# ---------------------------------------------------------------------------
_HEURISTIC_STRINGS: Dict[str, Dict[str, str]] = {
    "tr": {
        "heuristic_source": "Akıllı Dahili AI Danışmanı (Heuristic Advisor Engine)",
        "heuristic_fallback_reason": "API çağrısı başarısız oldu, dahili motora geçildi",
        "rating_high": "Yüksek müşteri memnuniyet puanı ({v}/5.0), sosyal kanıtı ve dönüşüm oranını doğrudan artırıyor.",
        "rating_low": "Nispeten düşük ürün puanı ({v}/5.0), organik sıralamayı ve dönüşümleri olumsuz etkileyebilir.",
        "price_high": "Yüksek birim fiyat (${v:,.2f}), fiyat odaklı segmentte satın alma bariyeri oluşturabilir.",
        "price_low": "Erişilebilir ve rekabetçi fiyatlandırma (${v:,.2f}) pazara giriş direncini kırıyor.",
        "discount_high": "Çekici indirim/promosyon stratejisi (%{v:.0f}), aciliyet hissi ve sepete ekleme motivasyonu yaratıyor.",
        "discount_opportunity": "Lansman dönemine özel %15-25 aralığında sınırlı süreli promosyon tanımlanabilir.",
        "reviews_high": "Güçlü inceleme hacmi ({v} değerlendirme), algılanan güveni maksimize ediyor.",
        "reviews_low": "Düşük inceleme sayısı ({v}), potansiyel müşterilerin güven bariyerine takılmasına neden olabilir.",
        "prime_yes": "Hızlı/Prime kargo rozeti sayesinde filtrelemelerde üst sıralarda yer alma avantajı.",
        "prime_no": "Prime / Hızlı kargo avantajı bulunmuyor; sepeti terk etme oranını artırabilir.",
        "video_yes": "Tanıtım videosu mevcut, ürünün kavranabilirliğini ve kalma süresini artırıyor.",
        "video_opportunity": "15-30 saniyelik dinamik bir ürün deneyim videosu eklenerek dönüşüm %35 artırılabilir.",
        "api_yes": "Geliştirici dostu API veya ücretsiz deneme paketi, büyüme hunisini genişletiyor.",
        "default_strength": "Ürünün hedef kitle tanımlaması ve kategori uyumu standart pazar kriterlerini karşılıyor.",
        "default_weakness": "Pazardaki agresif fiyat savaşı ve benzer ikame ürün bolluğu marj baskısı yaratabilir.",
        "default_opportunity": "Mikro-influencer iş birlikleri ve niş topluluklarda ürün lansmanı ile organik büyüme.",
        "default_threat": "Köklü pazar liderlerinin bütçe avantajı ve yeni giren rakiplerin benzer teklifleri.",
        "outlook_high": "oldukça yüksek bir başarı ve talep potansiyeli taşıdığını",
        "outlook_low": "kritik optimizasyonlar yapılmadan riskli bir pazar konumunda olduğunu",
        "report_template": """### {status_emoji} Yönetici Özeti & Başarı Değerlendirmesi
Modelimiz bu ürün için **%{probability_percent:.1f} başarı olasılığı** ve **"{prediction_label}"** tahmini üretmiştir.
Bu skor, ürünün mevcut pazar konfigürasyonunda {outlook} göstermektedir.

---

### 🔍 Kapsamlı Ürün SWOT Analizi

#### 🛡️ Güçlü Yönler (Strengths)
{strengths}

#### ⚠️ Zayıf Yönler & Riskler (Weaknesses)
{weaknesses}

#### 🌟 Fırsatlar & Büyüme Alanları (Opportunities)
{opportunities}

#### 🥊 Tehditler & Rekabet Baskısı (Threats)
{threats}

---

### 💡 Stratejik Aksiyon & Optimizasyon Tavsiyeleri

1. **🏷️ Fiyatlandırma & Paketleme Stratejisi**:
   - Fiyat esnekliğini test etmek için dinamik A/B fiyat testleri uygulayın.
   - Ürünün değer algısını artırmak için 'Bundle' (Paket) veya ücretsiz bonus dijital içerik ekleyin.

2. **📝 Başlık & Açıklama (SEO / Metin) İyileştirmesi**:
   - Ürün başlığında en çok aranan 3 anahtar kelimeyi (örn. *Ultra, Ergonomik, Pro*) ilk 40 karakterde konumlandırın.
   - Açıklama metninde özelliklerden çok 'kullanıcıya sağlanan faydayı' vurgulayın.

3. **📢 Pazarlama & Dönüşüm Optimizasyonu**:
   - Sosyal kanıtı güçlendirmek için ilk 30 günde erken kullanıcı inceleme programı başlatın.
   - Sosyal medya reklamlarında doğrudan ürün çözümünü gösteren 15 saniyelik dikey videolar kullanın.

---

### 🚀 90 Günlük Büyüme Yol Haritası
- **📅 Gün 1 - 30 (Lansman & Validasyon)**: Listeleme optimizasyonu, ilk 50 müşteri geri bildirimi ve A/B fiyat testi.
- **📅 Gün 31 - 60 (Ölçeklenme)**: Hedefli dijital reklamlar, influencer tanıtımları ve stok/ikmal optimizasyonu.
- **📅 Gün 61 - 90 (Sürdürülebilirlik)**: Tekrar satın alma kampanyaları, çapraz satış ve ürün varyant geliştirme.
""",
    },
    "en": {
        "heuristic_source": "Smart Built-in AI Advisor (Heuristic Advisor Engine)",
        "heuristic_fallback_reason": "API call failed, switched to the built-in engine",
        "rating_high": "High customer satisfaction score ({v}/5.0) directly boosts social proof and conversion rate.",
        "rating_low": "Relatively low product rating ({v}/5.0) may hurt organic ranking and conversions.",
        "price_high": "High unit price (${v:,.2f}) can create a purchase barrier in a price-sensitive segment.",
        "price_low": "Accessible, competitive pricing (${v:,.2f}) lowers the barrier to market entry.",
        "discount_high": "An attractive discount/promotion strategy ({v:.0f}%) creates urgency and add-to-cart motivation.",
        "discount_opportunity": "A time-limited 15-25% launch promotion could be defined.",
        "reviews_high": "Strong review volume ({v} reviews) maximizes perceived trust.",
        "reviews_low": "Low review count ({v}) may cause potential customers to hit a trust barrier.",
        "prime_yes": "The fast/Prime shipping badge gives an advantage in filtered search rankings.",
        "prime_no": "No Prime / fast-shipping advantage; this may raise the cart-abandonment rate.",
        "video_yes": "A promotional video is present, improving comprehension and dwell time.",
        "video_opportunity": "Adding a 15-30 second dynamic product demo video could lift conversion by ~35%.",
        "api_yes": "A developer-friendly API or free trial tier widens the growth funnel.",
        "default_strength": "The product's target audience definition and category fit meet standard market criteria.",
        "default_weakness": "Aggressive price competition and abundant similar substitute products may pressure margins.",
        "default_opportunity": "Organic growth via micro-influencer partnerships and niche community launches.",
        "default_threat": "Established market leaders' budget advantage and similar offers from new entrants.",
        "outlook_high": "carries a fairly high success and demand potential",
        "outlook_low": "sits in a risky market position without critical optimizations",
        "report_template": """### {status_emoji} Executive Summary & Success Assessment
Our model produced a **{probability_percent:.1f}% success probability** and a **"{prediction_label}"** prediction for this product.
This score indicates that, in its current market configuration, the product {outlook}.

---

### 🔍 Comprehensive Product SWOT Analysis

#### 🛡️ Strengths
{strengths}

#### ⚠️ Weaknesses & Risks
{weaknesses}

#### 🌟 Opportunities & Growth Areas
{opportunities}

#### 🥊 Threats & Competitive Pressure
{threats}

---

### 💡 Strategic Action & Optimization Recommendations

1. **🏷️ Pricing & Bundling Strategy**:
   - Run dynamic A/B price tests to probe price elasticity.
   - Add a bundle or a free bonus digital extra to increase perceived value.

2. **📝 Title & Description (SEO / Copy) Improvement**:
   - Place the 3 most-searched keywords (e.g. *Ultra, Ergonomic, Pro*) within the first 40 characters of the title.
   - Emphasize the benefit to the user rather than raw features in the description.

3. **📢 Marketing & Conversion Optimization**:
   - Launch an early-reviewer program in the first 30 days to strengthen social proof.
   - Use 15-second vertical videos in social ads that show the product solving the problem directly.

---

### 🚀 90-Day Growth Roadmap
- **📅 Day 1-30 (Launch & Validation)**: Listing optimization, first 50 customer feedback rounds, and A/B price testing.
- **📅 Day 31-60 (Scaling)**: Targeted digital ads, influencer promotions, and inventory/replenishment optimization.
- **📅 Day 61-90 (Sustainability)**: Repeat-purchase campaigns, cross-selling, and product variant development.
""",
    },
    "de": {
        "heuristic_source": "Intelligenter integrierter KI-Berater (Heuristik-Advisor-Engine)",
        "heuristic_fallback_reason": "API-Aufruf fehlgeschlagen, auf die integrierte Engine umgeschaltet",
        "rating_high": "Hohe Kundenzufriedenheit ({v}/5.0) steigert direkt Social Proof und Konversionsrate.",
        "rating_low": "Relativ niedrige Produktbewertung ({v}/5.0) kann das organische Ranking und die Konversionen beeinträchtigen.",
        "price_high": "Hoher Stückpreis (${v:,.2f}) kann in einem preissensiblen Segment eine Kaufbarriere darstellen.",
        "price_low": "Zugängliche, wettbewerbsfähige Preisgestaltung (${v:,.2f}) senkt die Eintrittsbarriere in den Markt.",
        "discount_high": "Eine attraktive Rabatt-/Promotionsstrategie ({v:.0f}%) erzeugt Dringlichkeit und Kaufmotivation.",
        "discount_opportunity": "Eine zeitlich begrenzte Launch-Promotion von 15-25% könnte definiert werden.",
        "reviews_high": "Starkes Bewertungsvolumen ({v} Bewertungen) maximiert das wahrgenommene Vertrauen.",
        "reviews_low": "Geringe Anzahl an Bewertungen ({v}) kann bei potenziellen Kunden eine Vertrauensbarriere erzeugen.",
        "prime_yes": "Das Schnell-/Prime-Versand-Abzeichen verschafft einen Vorteil bei gefilterten Suchrankings.",
        "prime_no": "Kein Prime-/Schnellversand-Vorteil vorhanden; dies kann die Warenkorbabbruchrate erhöhen.",
        "video_yes": "Ein Werbevideo ist vorhanden und verbessert Verständlichkeit sowie Verweildauer.",
        "video_opportunity": "Ein 15-30-sekündiges dynamisches Produktvideo könnte die Konversion um ca. 35% steigern.",
        "api_yes": "Eine entwicklerfreundliche API oder ein kostenloses Test-Tier erweitert den Wachstumstrichter.",
        "default_strength": "Die Zielgruppendefinition und die Kategoriepassung des Produkts erfüllen die Standardmarktkriterien.",
        "default_weakness": "Aggressiver Preiswettbewerb und zahlreiche ähnliche Ersatzprodukte können die Margen unter Druck setzen.",
        "default_opportunity": "Organisches Wachstum durch Mikro-Influencer-Kooperationen und Nischen-Community-Launches.",
        "default_threat": "Budgetvorteil etablierter Marktführer und ähnliche Angebote neuer Wettbewerber.",
        "outlook_high": "in seiner aktuellen Marktkonfiguration ein recht hohes Erfolgs- und Nachfragepotenzial hat",
        "outlook_low": "sich ohne kritische Optimierungen in einer riskanten Marktposition befindet",
        "report_template": """### {status_emoji} Zusammenfassung & Erfolgsbewertung
Unser Modell hat für dieses Produkt eine **Erfolgswahrscheinlichkeit von {probability_percent:.1f}%** und die Vorhersage **„{prediction_label}“** ermittelt.
Dieser Wert zeigt, dass das Produkt in seiner aktuellen Marktkonfiguration {outlook}.

---

### 🔍 Umfassende Produkt-SWOT-Analyse

#### 🛡️ Stärken
{strengths}

#### ⚠️ Schwächen & Risiken
{weaknesses}

#### 🌟 Chancen & Wachstumsfelder
{opportunities}

#### 🥊 Bedrohungen & Wettbewerbsdruck
{threats}

---

### 💡 Strategische Maßnahmen & Optimierungsempfehlungen

1. **🏷️ Preis- & Bündelungsstrategie**:
   - Führen Sie dynamische A/B-Preistests durch, um die Preiselastizität zu prüfen.
   - Fügen Sie ein Bundle oder einen kostenlosen digitalen Bonus hinzu, um den wahrgenommenen Wert zu steigern.

2. **📝 Titel- & Beschreibungsverbesserung (SEO / Text)**:
   - Platzieren Sie die 3 meistgesuchten Schlüsselwörter (z. B. *Ultra, Ergonomisch, Pro*) in den ersten 40 Zeichen des Titels.
   - Betonen Sie in der Beschreibung eher den Nutzen für den Kunden als reine Funktionen.

3. **📢 Marketing- & Conversion-Optimierung**:
   - Starten Sie in den ersten 30 Tagen ein Frühbewerter-Programm, um den Social Proof zu stärken.
   - Verwenden Sie in Social-Media-Anzeigen 15-sekündige vertikale Videos, die die Produktlösung direkt zeigen.

---

### 🚀 90-Tage-Wachstumsfahrplan
- **📅 Tag 1-30 (Markteinführung & Validierung)**: Listing-Optimierung, Feedback der ersten 50 Kunden und A/B-Preistests.
- **📅 Tag 31-60 (Skalierung)**: Gezielte digitale Anzeigen, Influencer-Promotionen und Bestands-/Nachschuboptimierung.
- **📅 Tag 61-90 (Nachhaltigkeit)**: Wiederkaufkampagnen, Cross-Selling und Entwicklung von Produktvarianten.
""",
    },
}
