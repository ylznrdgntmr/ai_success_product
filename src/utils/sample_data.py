"""
Sample Dataset Generator & Provider
Provides high-quality realistic datasets for testing and demonstration.
"""

import numpy as np
import pandas as pd


def generate_ecommerce_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic E-Commerce Product Success dataset."""
    np.random.seed(random_state)

    categories = [
        "Electronics", "Home & Kitchen", "Beauty & Personal Care",
        "Sports & Outdoors", "Fashion & Apparel", "Toys & Games",
        "Books & Media", "Automotive", "Health & Wellness"
    ]

    title_adjectives = [
        "Premium", "Ultra", "Wireless", "Smart", "Ergonomic", "Portable",
        "Durable", "Eco-Friendly", "Pro", "Compact", "High-Performance",
        "Multi-Functional", "Classic", "Advanced", "Noise-Cancelling"
    ]

    title_nouns = [
        "Headphones", "Coffee Maker", "Yoga Mat", "Smartwatch", "Backpack",
        "Air Fryer", "Bluetooth Speaker", "Water Bottle", "Running Shoes",
        "Desk Lamp", "Power Bank", "Mechanical Keyboard", "Skin Serum", "Massage Gun"
    ]

    titles = [
        f"{np.random.choice(title_adjectives)} {np.random.choice(title_nouns)} with {np.random.choice(['Fast Charge', 'HD Audio', 'All-Day Battery', 'Memory Foam', 'Touch Control', 'Stainless Steel', 'Organic Extract'])}"
        for _ in range(n_samples)
    ]

    cat_choice = np.random.choice(categories, size=n_samples)
    
    # Base prices per category
    base_prices = {
        "Electronics": 89.99, "Home & Kitchen": 49.99, "Beauty & Personal Care": 24.99,
        "Sports & Outdoors": 34.99, "Fashion & Apparel": 39.99, "Toys & Games": 19.99,
        "Books & Media": 14.99, "Automotive": 45.00, "Health & Wellness": 29.99
    }

    prices = []
    list_prices = []
    for cat in cat_choice:
        base = base_prices[cat]
        price = np.round(np.random.gamma(shape=3.0, scale=base / 3.0) + 5.0, 2)
        discount_ratio = np.random.choice([1.0, 1.15, 1.25, 1.4, 1.6], p=[0.25, 0.3, 0.25, 0.15, 0.05])
        list_price = np.round(price * discount_ratio, 2)
        prices.append(price)
        list_prices.append(list_price)

    ratings = np.clip(np.round(np.random.normal(loc=4.1, scale=0.5, size=n_samples), 1), 1.0, 5.0)
    reviews_count = np.random.exponential(scale=350, size=n_samples).astype(int) + 5
    has_prime = np.random.choice([True, False], size=n_samples, p=[0.75, 0.25])
    has_video = np.random.choice([True, False], size=n_samples, p=[0.45, 0.55])
    images_count = np.random.randint(1, 9, size=n_samples)
    description_length = np.random.normal(loc=450, scale=120, size=n_samples).astype(int)
    description_length = np.clip(description_length, 50, 1500)

    # Success probability logic (Realistic ground truth with noise)
    price_arr = np.array(prices)
    list_price_arr = np.array(list_prices)
    discount_pct = (list_price_arr - price_arr) / (list_price_arr + 1e-5) * 100

    score = (
        (ratings - 3.5) * 1.8 +
        np.log1p(reviews_count) * 0.45 +
        (discount_pct / 20.0) * 0.35 +
        has_prime.astype(int) * 0.8 +
        has_video.astype(int) * 0.5 +
        (images_count >= 5).astype(int) * 0.4 -
        (price_arr / 120.0) * 0.3 +
        np.random.normal(0, 0.5, size=n_samples)
    )

    threshold = np.percentile(score, 65) # Top 35% successful
    is_success = (score >= threshold).astype(int)
    monthly_sales = np.clip(np.round(np.exp(score * 0.8 + 3.0)).astype(int), 10, 15000)

    df = pd.DataFrame({
        "product_id": [f"PROD-{1000 + i}" for i in range(n_samples)],
        "title": titles,
        "category": cat_choice,
        "price": prices,
        "list_price": list_prices,
        "discount_percent": np.round(discount_pct, 1),
        "rating": ratings,
        "reviews_count": reviews_count,
        "has_prime_badge": has_prime,
        "has_product_video": has_video,
        "images_count": images_count,
        "description_char_length": description_length,
        "monthly_sales_units": monthly_sales,
        "is_best_seller": is_success
    })

    return df


def generate_saas_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic SaaS Product Launch Success dataset."""
    np.random.seed(random_state)

    target_audiences = ["B2B Enterprise", "SMB & Startups", "Freelancers & Creators", "B2C Consumers", "Developers / Tech Teams"]
    pricing_models = ["Freemium", "Usage-Based", "Tiered Monthly", "Flat Rate", "Free Trial + Paid"]
    launch_platforms = ["Product Hunt", "Hacker News", "AppSumo", "Direct Inbound / SEO", "LinkedIn Ads", "Cold Outreach"]

    names = [f"CloudApp-{i+1:03d}" for i in range(n_samples)]
    audiences = np.random.choice(target_audiences, size=n_samples)
    models = np.random.choice(pricing_models, size=n_samples)
    platforms = np.random.choice(launch_platforms, size=n_samples)

    monthly_prices = np.random.choice([0.0, 9.99, 19.99, 49.0, 99.0, 199.0, 499.0], size=n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.07, 0.03])
    free_tier = (monthly_prices == 0.0) | np.random.choice([True, False], size=n_samples, p=[0.4, 0.6])
    features_count = np.random.randint(3, 25, size=n_samples)
    integrations_count = np.random.randint(0, 30, size=n_samples)
    team_size = np.random.choice([1, 2, 4, 8, 15, 30], size=n_samples, p=[0.3, 0.25, 0.25, 0.12, 0.06, 0.02])
    marketing_budget = np.random.choice([0, 500, 2000, 5000, 15000, 50000], size=n_samples, p=[0.25, 0.3, 0.2, 0.15, 0.07, 0.03])
    has_api = np.random.choice([True, False], size=n_samples, p=[0.6, 0.4])
    onboarding_time_mins = np.random.randint(1, 45, size=n_samples)

    # Success probability
    score = (
        (integrations_count / 10.0) * 0.6 +
        (features_count / 8.0) * 0.4 +
        np.log1p(marketing_budget) * 0.35 +
        (team_size >= 3).astype(int) * 0.5 +
        has_api.astype(int) * 0.4 -
        (onboarding_time_mins / 20.0) * 0.5 +
        (platforms == "Product Hunt").astype(int) * 0.3 +
        np.random.normal(0, 0.6, size=n_samples)
    )

    threshold = np.percentile(score, 70)
    is_successful_launch = (score >= threshold).astype(int)
    mrr_usd = np.clip(np.round(np.exp(score * 0.7 + 6.5)).astype(int), 0, 85000)

    return pd.DataFrame({
        "saas_name": names,
        "target_audience": audiences,
        "pricing_model": models,
        "monthly_price_usd": monthly_prices,
        "free_tier_available": free_tier,
        "features_count": features_count,
        "integrations_count": integrations_count,
        "team_size": team_size,
        "marketing_budget_usd": marketing_budget,
        "launch_platform": platforms,
        "has_public_api": has_api,
        "onboarding_time_mins": onboarding_time_mins,
        "mrr_month_6_usd": mrr_usd,
        "is_successful_launch": is_successful_launch
    })


def generate_crowdfunding_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic Kickstarter / Crowdfunding Success dataset."""
    np.random.seed(random_state)

    categories = ["Technology", "Design", "Games", "Film & Video", "Music", "Food & Craft", "Publishing", "Fashion"]
    cat_choice = np.random.choice(categories, size=n_samples)

    goals = np.random.choice([2500, 5000, 10000, 25000, 50000, 100000, 250000], size=n_samples, p=[0.15, 0.25, 0.25, 0.18, 0.1, 0.05, 0.02])
    duration_days = np.random.choice([15, 30, 45, 60], size=n_samples, p=[0.1, 0.6, 0.2, 0.1])
    has_video = np.random.choice([True, False], size=n_samples, p=[0.7, 0.3])
    reward_tiers_count = np.random.randint(1, 12, size=n_samples)
    faq_count = np.random.randint(0, 15, size=n_samples)
    updates_count = np.random.randint(0, 25, size=n_samples)
    early_bird_discount = np.random.choice([True, False], size=n_samples, p=[0.55, 0.45])
    creator_prior_projects = np.random.choice([0, 1, 2, 4, 8], size=n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])

    score = (
        has_video.astype(int) * 0.9 +
        (creator_prior_projects > 0).astype(int) * 0.8 +
        (reward_tiers_count >= 4).astype(int) * 0.4 +
        (faq_count / 5.0) * 0.3 +
        (updates_count / 6.0) * 0.5 +
        early_bird_discount.astype(int) * 0.4 -
        np.log1p(goals) * 0.4 +
        np.random.normal(0, 0.55, size=n_samples)
    )

    threshold = np.percentile(score, 60)
    is_funded = (score >= threshold).astype(int)
    backers = np.clip(np.round(np.exp(score * 0.6 + 4.5)).astype(int), 5, 8000)
    pledged_usd = np.clip(np.round(backers * np.random.uniform(25, 120, size=n_samples)), 100, 1500000)

    return pd.DataFrame({
        "project_id": [f"KS-{2000 + i}" for i in range(n_samples)],
        "category": cat_choice,
        "funding_goal_usd": goals,
        "duration_days": duration_days,
        "has_pitch_video": has_video,
        "reward_tiers_count": reward_tiers_count,
        "faq_count": faq_count,
        "project_updates_count": updates_count,
        "early_bird_discount": early_bird_discount,
        "creator_prior_projects": creator_prior_projects,
        "total_backers": backers,
        "pledged_amount_usd": pledged_usd,
        "is_successfully_funded": is_funded
    })


def generate_mobile_app_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic Mobile App Store Success dataset."""
    np.random.seed(random_state)

    categories = ["Productivity", "Games", "Health & Fitness", "Finance", "Education",
                  "Social Networking", "Photo & Video", "Lifestyle", "Utilities"]
    monetization_models = ["Free (Ads)", "Freemium (IAP)", "Paid Upfront", "Subscription"]

    app_names = [f"App-{i+1:04d}" for i in range(n_samples)]
    cat_choice = np.random.choice(categories, size=n_samples)
    monetization = np.random.choice(monetization_models, size=n_samples, p=[0.35, 0.35, 0.1, 0.2])

    app_size_mb = np.clip(np.random.gamma(shape=2.5, scale=35, size=n_samples), 5, 900)
    onboarding_screens = np.random.randint(0, 8, size=n_samples)
    permissions_requested = np.random.randint(0, 12, size=n_samples)
    has_push_notifications = np.random.choice([True, False], size=n_samples, p=[0.65, 0.35])
    has_dark_mode = np.random.choice([True, False], size=n_samples, p=[0.55, 0.45])
    update_frequency_days = np.random.choice([7, 14, 30, 60, 120], size=n_samples, p=[0.15, 0.25, 0.35, 0.15, 0.1])
    rating = np.clip(np.round(np.random.normal(loc=4.0, scale=0.5, size=n_samples), 1), 1.0, 5.0)
    rating_count = np.random.exponential(scale=800, size=n_samples).astype(int) + 5
    ad_spend_usd = np.random.choice([0, 500, 2500, 10000, 50000], size=n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])

    score = (
        (rating - 3.5) * 1.6 +
        np.log1p(rating_count) * 0.4 +
        has_push_notifications.astype(int) * 0.35 +
        has_dark_mode.astype(int) * 0.15 +
        (onboarding_screens <= 3).astype(int) * 0.4 -
        (permissions_requested / 6.0) * 0.4 -
        (update_frequency_days / 60.0) * 0.3 +
        np.log1p(ad_spend_usd) * 0.25 +
        np.random.normal(0, 0.55, size=n_samples)
    )

    threshold = np.percentile(score, 65)
    is_top_chart = (score >= threshold).astype(int)
    monthly_downloads = np.clip(np.round(np.exp(score * 0.75 + 5.0)).astype(int), 20, 500000)

    return pd.DataFrame({
        "app_name": app_names,
        "category": cat_choice,
        "monetization_model": monetization,
        "app_size_mb": np.round(app_size_mb, 1),
        "onboarding_screens": onboarding_screens,
        "permissions_requested": permissions_requested,
        "has_push_notifications": has_push_notifications,
        "has_dark_mode": has_dark_mode,
        "update_frequency_days": update_frequency_days,
        "rating": rating,
        "rating_count": rating_count,
        "ad_spend_usd": ad_spend_usd,
        "monthly_downloads": monthly_downloads,
        "is_top_chart_app": is_top_chart
    })


def generate_online_course_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic Online Course / Digital Info-Product Success dataset."""
    np.random.seed(random_state)

    subjects = ["Programming", "Business & Marketing", "Design", "Personal Finance",
                "Photography", "Music", "Language Learning", "Health & Wellness"]
    formats = ["Video Lectures", "Cohort-Based Live", "Text + Video Hybrid", "Workshop Series"]

    titles = [f"Complete {np.random.choice(subjects)} Masterclass {i+1}" for i in range(n_samples)]
    subject_choice = np.random.choice(subjects, size=n_samples)
    format_choice = np.random.choice(formats, size=n_samples)

    price = np.round(np.random.choice([0, 19, 49, 99, 199, 499, 999], size=n_samples,
                                       p=[0.05, 0.2, 0.25, 0.2, 0.15, 0.1, 0.05]), 2)
    total_duration_hours = np.clip(np.random.gamma(shape=2.0, scale=6.0, size=n_samples), 0.5, 80)
    lecture_count = np.random.randint(3, 150, size=n_samples)
    has_certificate = np.random.choice([True, False], size=n_samples, p=[0.6, 0.4])
    has_community_access = np.random.choice([True, False], size=n_samples, p=[0.45, 0.55])
    instructor_prior_courses = np.random.choice([0, 1, 2, 5, 10, 20], size=n_samples, p=[0.35, 0.25, 0.15, 0.15, 0.07, 0.03])
    rating = np.clip(np.round(np.random.normal(loc=4.3, scale=0.4, size=n_samples), 1), 1.0, 5.0)
    reviews_count = np.random.exponential(scale=150, size=n_samples).astype(int) + 2
    completion_rate_pct = np.clip(np.random.normal(loc=45, scale=20, size=n_samples), 2, 98)

    score = (
        (rating - 3.8) * 1.7 +
        np.log1p(reviews_count) * 0.5 +
        has_certificate.astype(int) * 0.35 +
        has_community_access.astype(int) * 0.4 +
        (instructor_prior_courses > 1).astype(int) * 0.5 +
        (completion_rate_pct / 40.0) * 0.4 -
        np.log1p(price) * 0.25 +
        np.random.normal(0, 0.5, size=n_samples)
    )

    threshold = np.percentile(score, 65)
    is_bestselling = (score >= threshold).astype(int)
    enrollments = np.clip(np.round(np.exp(score * 0.7 + 4.5)).astype(int), 10, 120000)

    return pd.DataFrame({
        "course_title": titles,
        "subject": subject_choice,
        "format": format_choice,
        "price_usd": price,
        "total_duration_hours": np.round(total_duration_hours, 1),
        "lecture_count": lecture_count,
        "has_certificate": has_certificate,
        "has_community_access": has_community_access,
        "instructor_prior_courses": instructor_prior_courses,
        "rating": rating,
        "reviews_count": reviews_count,
        "completion_rate_pct": np.round(completion_rate_pct, 1),
        "total_enrollments": enrollments,
        "is_bestselling_course": is_bestselling
    })


SAMPLE_DATASETS = {
    "🛒 E-Ticaret Ürün Başarısı (Amazon Ürünleri)": {
        "generator": generate_ecommerce_dataset,
        "default_target": "is_best_seller",
        "task_type": "classification",
        "description": "Fiyat, indirim, puan, inceleme sayısı, görsel sayısı gibi özelliklerle ürünün Çok Satan (Best Seller) olma başarısını tahmin eder."
    },
    "💻 SaaS & Startup Lansman Başarısı": {
        "generator": generate_saas_dataset,
        "default_target": "is_successful_launch",
        "task_type": "classification",
        "description": "Fiyatlandırma modeli, entegrasyon sayısı, ekip büyüklüğü ve pazarlama bütçesiyle yeni SaaS ürününün karlı bir lansman yapma başarısını tahmin eder."
    },
    "🚀 Kitle Fonlaması & Kickstarter Başarısı": {
        "generator": generate_crowdfunding_dataset,
        "default_target": "is_successfully_funded",
        "task_type": "classification",
        "description": "Hedef fon, kampanya süresi, video varlığı, ödül kademeleri ve güncelleme sıklığıyla projenin fonlanma başarısını tahmin eder."
    },
    "📱 Mobil Uygulama Başarısı (App Store)": {
        "generator": generate_mobile_app_dataset,
        "default_target": "is_top_chart_app",
        "task_type": "classification",
        "description": "Kategori, para kazanma modeli, boyut, izin sayısı, puan ve reklam bütçesiyle bir mobil uygulamanın popüler listelere girme başarısını tahmin eder."
    },
    "🎓 Online Kurs & Dijital Eğitim Başarısı": {
        "generator": generate_online_course_dataset,
        "default_target": "is_bestselling_course",
        "task_type": "classification",
        "description": "Fiyat, süre, sertifika, topluluk erişimi, eğitmen tecrübesi ve tamamlanma oranıyla bir online kursun çok satan olma başarısını tahmin eder."
    }
}
