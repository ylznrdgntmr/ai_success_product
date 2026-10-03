"""
Prompt Templates for the LLM AI Product Success Advisor.
Each template is provided in Turkish (tr), English (en) and German (de) so
that a live LLM call answers in the user's currently selected UI language.
"""

from typing import Dict

DEFAULT_LANG = "tr"


def _pick(templates: Dict[str, str], lang: str) -> str:
    return templates.get(lang, templates.get(DEFAULT_LANG))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_ADVISOR = {
    "tr": """Sen kıdemli bir Ürün Yöneticisi (Principal Product Manager), E-Ticaret & SaaS Büyüme Stratejisti ve Yapay Zeka Veri Bilim Proje Danışmanısın.

Görevin:
- Makine öğrenmesi modelinin tahmin sonuçlarını, ürün özelliklerini ve pazar dinamiklerini analiz ederek kullanıcıya **derinlemesine, somut, uygulanabilir** stratejik ürün başarı analizleri sunmak.
- Her analizi ürün başarı oranına göre uyarlamak: düşük başarı olasılığında **uyarı odaklı**, yüksek olasılıkta **büyüme ve ölçekleme odaklı** yaklaş.
- Tahmin edilen başarı oranını körü körüne kabul etme; tutarsız özellik kombinasyonlarına dikkat çek.

Yanıtlarını:
✅ Profesyonel, organize ve Türkçe olarak sun.
✅ Markdown başlıkları, emojiler, tablolar ve vurgular kullan.
✅ Somut rakamlar ve örnekler içeren aksiyon tavsiyeleri ver.
✅ Her bölümü ayrı bir başlık altında net şekilde ayır.
❌ Belirsiz, genel veya klişe cevaplar verme.
""",
    "en": """You are a senior Principal Product Manager, E-Commerce & SaaS Growth Strategist, and AI Data Science Project Advisor.

Your task:
- Analyze the machine learning model's prediction, the product's features, and market dynamics to give the user an **in-depth, concrete, actionable** strategic product-success analysis.
- Adapt each analysis to the success probability: **warning-focused** for low probability, **growth-and-scaling-focused** for high probability.
- Don't blindly accept the predicted success rate; call out inconsistent feature combinations when you see them.

Your response should:
✅ Be professional, well organized, and written in English.
✅ Use Markdown headings, emoji, tables and emphasis.
✅ Give action recommendations with concrete numbers and examples.
✅ Clearly separate each section under its own heading.
❌ Never give vague, generic, or cliché answers.
""",
    "de": """Du bist ein erfahrener Principal Product Manager, E-Commerce- & SaaS-Wachstumsstratege und KI-Datenwissenschafts-Projektberater.

Deine Aufgabe:
- Analysiere die Vorhersage des Machine-Learning-Modells, die Produktmerkmale und die Marktdynamik, um dem Nutzer eine **tiefgehende, konkrete, umsetzbare** strategische Produkterfolgsanalyse zu liefern.
- Passe jede Analyse an die Erfolgswahrscheinlichkeit an: bei geringer Wahrscheinlichkeit **warnungsorientiert**, bei hoher Wahrscheinlichkeit **wachstums- und skalierungsorientiert**.
- Akzeptiere die vorhergesagte Erfolgsquote nicht blind; weise auf widersprüchliche Merkmalskombinationen hin.

Deine Antwort soll:
✅ Professionell, gut strukturiert und auf Deutsch verfasst sein.
✅ Markdown-Überschriften, Emojis, Tabellen und Hervorhebungen verwenden.
✅ Handlungsempfehlungen mit konkreten Zahlen und Beispielen geben.
✅ Jeden Abschnitt klar unter einer eigenen Überschrift trennen.
❌ Niemals vage, allgemeine oder klischeehafte Antworten geben.
""",
}

# ---------------------------------------------------------------------------
# Main product-analysis user prompt
# ---------------------------------------------------------------------------
_PRODUCT_ANALYSIS_PROMPT_TEMPLATE = {
    "tr": """Aşağıda bir ürünün özellikleri, eğitilen Makine Öğrenmesi modelinin başarı tahmini ve en etkili faktörleri verilmiştir.

### 📋 ÜRÜN BİLGİLERİ VE ÖZELLİKLERİ:
{product_features}

### 🎯 MODEL TAHMİN SONUCU:
- **Tahmin Edilen Başarı Durumu:** {prediction_label}
- **Başarı Olasılığı / Skoru:** %{probability_percent:.1f}
- **En İyi AutoML Model:** {best_model_name} ({model_metric})

### ⚖️ MODELİN EN ETKİLİ KARAR FAKTÖRLERİ:
{top_drivers}

---

Lütfen bu ürünü **kapsamlı ve kişiselleştirilmiş** şekilde analiz et. Raporun aşağıdaki bölümleri içermeli:

## 📊 1. Yönetici Özeti & Başarı Değerlendirmesi
- Modelin verdiği **%{probability_percent:.1f}** oranının ne anlama geldiği ve pazar karşılığı.
- Bu ürün bu başarı seviyesinde hangi aşamadadır? (Lansman öncesi iyileştirme mi, ölçekleme mi, pivot mi?)
- Modelin kararında en belirleyici 2-3 özelliği özetle.

## 🔍 2. SWOT Analizi
Ürünün verilen özelliklerine göre spesifik ve gerçekçi SWOT çıkar:
### 🛡️ Güçlü Yönler (Strengths)
### ⚠️ Zayıf Yönler & Riskler (Weaknesses)
### 🌟 Fırsatlar & Büyüme Alanları (Opportunities)
### 🥊 Tehditler & Rekabet Baskısı (Threats)

## 💡 3. Somut Aksiyon & Optimizasyon Tavsiyeleri
Her tavsiye için **beklenen etki (%)** belirt:
- **🏷️ Fiyatlandırma Stratejisi**: Somut bir fiyat aralığı öner.
- **📝 Ürün Başlığı & Açıklama Optimizasyonu**: Güçlendirilmiş başlık örneği yaz.
- **📢 Pazarlama & Dönüşüm Taktikleri**: En etkili 2 kanal ve mesaj stratejisi.
- **⭐ İnceleme & Güven İnşası**: Hızlı sonuç verecek review/sosyal kanıt taktikleri.

## 🚀 4. 90 Günlük Büyüme Yol Haritası
| Dönem | Hedef | Aksiyon | Beklenen Sonuç |
|-------|-------|---------|----------------|
| Gün 1-30 (Temel) | ... | ... | ... |
| Gün 31-60 (Büyüme) | ... | ... | ... |
| Gün 61-90 (Ölçekleme) | ... | ... | ... |

## ⚡ 5. Hızlı Kazanımlar (Quick Wins)
Bu hafta yapılabilecek **3 somut** değişiklik nedir?
""",
    "en": """Below are a product's features, the trained Machine Learning model's success prediction, and the most influential factors.

### 📋 PRODUCT INFORMATION AND FEATURES:
{product_features}

### 🎯 MODEL PREDICTION RESULT:
- **Predicted Success Status:** {prediction_label}
- **Success Probability / Score:** {probability_percent:.1f}%
- **Best AutoML Model:** {best_model_name} ({model_metric})

### ⚖️ MODEL'S MOST INFLUENTIAL DECISION FACTORS:
{top_drivers}

---

Please analyze this product in a **comprehensive and personalized** way. The report must include these sections:

## 📊 1. Executive Summary & Success Assessment
- What the model's **{probability_percent:.1f}%** figure means and its real-world market implication.
- At what stage is this product given this success level? (Pre-launch improvement, scaling, or pivot?)
- Summarize the 2-3 most decisive features behind the model's decision.

## 🔍 2. SWOT Analysis
Derive a specific, realistic SWOT based on the given product features:
### 🛡️ Strengths
### ⚠️ Weaknesses & Risks
### 🌟 Opportunities & Growth Areas
### 🥊 Threats & Competitive Pressure

## 💡 3. Concrete Action & Optimization Recommendations
State the **expected impact (%)** for each recommendation:
- **🏷️ Pricing Strategy**: Suggest a concrete price range.
- **📝 Title & Description Optimization**: Write a strengthened title example.
- **📢 Marketing & Conversion Tactics**: The 2 most effective channels and messaging strategy.
- **⭐ Reviews & Trust Building**: Fast-payoff review / social-proof tactics.

## 🚀 4. 90-Day Growth Roadmap
| Period | Goal | Action | Expected Result |
|--------|------|--------|------------------|
| Day 1-30 (Foundation) | ... | ... | ... |
| Day 31-60 (Growth) | ... | ... | ... |
| Day 61-90 (Scaling) | ... | ... | ... |

## ⚡ 5. Quick Wins
What are **3 concrete** changes that could be made this week?
""",
    "de": """Im Folgenden finden Sie die Merkmale eines Produkts, die Erfolgsvorhersage des trainierten Machine-Learning-Modells und die einflussreichsten Faktoren.

### 📋 PRODUKTINFORMATIONEN UND MERKMALE:
{product_features}

### 🎯 MODELLVORHERSAGE:
- **Vorhergesagter Erfolgsstatus:** {prediction_label}
- **Erfolgswahrscheinlichkeit / Score:** {probability_percent:.1f}%
- **Bestes AutoML-Modell:** {best_model_name} ({model_metric})

### ⚖️ EINFLUSSREICHSTE ENTSCHEIDUNGSFAKTOREN DES MODELLS:
{top_drivers}

---

Bitte analysiere dieses Produkt **umfassend und personalisiert**. Der Bericht muss folgende Abschnitte enthalten:

## 📊 1. Zusammenfassung & Erfolgsbewertung
- Was die vom Modell ermittelte Quote von **{probability_percent:.1f}%** bedeutet und was sie am Markt aussagt.
- In welcher Phase befindet sich dieses Produkt bei diesem Erfolgsniveau? (Verbesserung vor der Markteinführung, Skalierung oder Pivot?)
- Fasse die 2-3 entscheidendsten Merkmale hinter der Modellentscheidung zusammen.

## 🔍 2. SWOT-Analyse
Leite eine spezifische, realistische SWOT-Analyse anhand der gegebenen Produktmerkmale ab:
### 🛡️ Stärken
### ⚠️ Schwächen & Risiken
### 🌟 Chancen & Wachstumsfelder
### 🥊 Bedrohungen & Wettbewerbsdruck

## 💡 3. Konkrete Maßnahmen & Optimierungsempfehlungen
Gib für jede Empfehlung die **erwartete Wirkung (%)** an:
- **🏷️ Preisstrategie**: Schlage eine konkrete Preisspanne vor.
- **📝 Titel- & Beschreibungsoptimierung**: Schreibe ein verbessertes Titelbeispiel.
- **📢 Marketing- & Conversion-Taktiken**: Die 2 wirksamsten Kanäle und die Botschaftsstrategie.
- **⭐ Bewertungen & Vertrauensaufbau**: Review-/Social-Proof-Taktiken mit schnellem Ergebnis.

## 🚀 4. 90-Tage-Wachstumsfahrplan
| Zeitraum | Ziel | Maßnahme | Erwartetes Ergebnis |
|----------|------|----------|----------------------|
| Tag 1-30 (Grundlage) | ... | ... | ... |
| Tag 31-60 (Wachstum) | ... | ... | ... |
| Tag 61-90 (Skalierung) | ... | ... | ... |

## ⚡ 5. Quick Wins
Welche **3 konkreten** Änderungen könnten diese Woche vorgenommen werden?
""",
}

_DATASET_INSIGHTS_PROMPT_TEMPLATE = {
    "tr": """Aşağıda veri setinin genel profili ve eğitilen Makine Öğrenmesi modellerinin özellikleri verilmiştir:

- Veri Seti Satır Sayısı: {n_rows}
- Sütun Sayısı: {n_cols}
- Hedef Değişken: {target_col} ({task_type})
- En İyi Model: {best_model_name}
- En Etkili 5 Özellik (Top Feature Importances): {top_features}

Bu sektörel veriyi bir Ürün Direktörü gözüyle analiz ederek işletmelere ve girişimcilere bu kategoride başarılı ürün çıkarmaları için 4 kritik stratejik kural çıkar.
""",
    "en": """Below is the dataset's general profile and the trained Machine Learning models' characteristics:

- Dataset Row Count: {n_rows}
- Column Count: {n_cols}
- Target Variable: {target_col} ({task_type})
- Best Model: {best_model_name}
- Top 5 Feature Importances: {top_features}

Analyze this sector's data from the perspective of a Head of Product and derive 4 critical strategic rules for businesses and founders to build a successful product in this category.
""",
    "de": """Im Folgenden finden Sie das allgemeine Profil des Datensatzes und die Merkmale der trainierten Machine-Learning-Modelle:

- Anzahl der Zeilen im Datensatz: {n_rows}
- Anzahl der Spalten: {n_cols}
- Zielvariable: {target_col} ({task_type})
- Bestes Modell: {best_model_name}
- Top 5 wichtigste Merkmale: {top_features}

Analysiere diese Branchendaten aus der Perspektive eines Head of Product und leite 4 kritische strategische Regeln für Unternehmen und Gründer ab, um in dieser Kategorie ein erfolgreiches Produkt zu entwickeln.
""",
}


def get_system_prompt(lang: str = DEFAULT_LANG) -> str:
    return _pick(_SYSTEM_PROMPT_ADVISOR, lang)


def get_product_analysis_prompt(lang: str = DEFAULT_LANG, **kwargs) -> str:
    return _pick(_PRODUCT_ANALYSIS_PROMPT_TEMPLATE, lang).format(**kwargs)


def get_dataset_insights_prompt(lang: str = DEFAULT_LANG, **kwargs) -> str:
    return _pick(_DATASET_INSIGHTS_PROMPT_TEMPLATE, lang).format(**kwargs)


# ---------------------------------------------------------------------------
# Backwards-compatible module-level constants (Turkish), kept so any external
# code importing the old names directly still works.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_ADVISOR = _SYSTEM_PROMPT_ADVISOR["tr"]
PRODUCT_ANALYSIS_PROMPT_TEMPLATE = _PRODUCT_ANALYSIS_PROMPT_TEMPLATE["tr"]
DATASET_INSIGHTS_PROMPT_TEMPLATE = _DATASET_INSIGHTS_PROMPT_TEMPLATE["tr"]
