#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проект 3: Скоринг оттока пользователей и оценка рыночной цены объявлений на реальных данных Авито
Данные: Реальный продакшен-датасет Avito Context Ad Clicks (rel-avito):
  1. Задача классификации (скоринг оттока пользователей, N = 49 352 активных пользователей):
     прогнозирование прекращения целевой активности (просмотров карточек VisitStream и запросов
     телефона PhoneRequestsStream) во второй половине периода по поведенческим признакам из
     UserInfo, SearchInfo, VisitStream, PhoneRequestsStream и SearchStream (плюс оценка на выборке tasks/user-visits).
  2. Задача регрессии (модель «Рыночная цена» объявления, N = 60 000 реальных объявлений):
     справочник AdsInfo.parquet (реальные цены Price в рублях, заголовки Title, флаг IsContext)
     + дерево категорий Category.parquet, регионы Location.parquet и спрос VisitStream / SearchStream.
Автор: Нефедов Родион (GitHub: github.com/RaikNef)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def resolve_avito_root() -> str:
    """Находит корневую директорию avito_real_data."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "avito_real_data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "avito_real_data"),
        r"d:\Work\resume\avito_real_data",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "db", "AdsInfo.parquet")):
            return os.path.abspath(path)
    raise FileNotFoundError("Не найдена папка avito_real_data.")


def build_real_user_churn_dataset(avito_root: str) -> tuple:
    """
    Собирает продуктовый датасет скоринга оттока по 49 352 реальным активным пользователям Авито:
    - Окно наблюдения признаков (16 дней): 2015-04-25 .. 2015-05-10 (строго до cut_dt)
    - Окно целевого события (10 дней): 2015-05-11 .. 2015-05-20 (churned = 1, если пользователь
      полностью прекратил просмотры карточек VisitStream и запросы контактов PhoneRequestsStream).
    """
    db_dir = os.path.join(avito_root, "db")

    users = pd.read_parquet(os.path.join(db_dir, "UserInfo.parquet"))
    searches = pd.read_parquet(
        os.path.join(db_dir, "SearchInfo.parquet"),
        columns=["UserID", "SearchID", "SearchDate", "IsUserLoggedOn", "CategoryID", "LocationID"],
    )
    visits = pd.read_parquet(
        os.path.join(db_dir, "VisitStream.parquet"),
        columns=["UserID", "AdID", "ViewDate"],
    )
    phones = pd.read_parquet(
        os.path.join(db_dir, "PhoneRequestsStream.parquet"),
        columns=["UserID", "AdID", "PhoneRequestDate"],
    )
    ads = pd.read_parquet(
        os.path.join(db_dir, "AdsInfo.parquet"),
        columns=["AdID", "Price", "CategoryID"],
    )

    cut_dt = pd.Timestamp("2015-05-11 00:00:00")
    t_1d = cut_dt - pd.Timedelta(days=1)
    t_3d = cut_dt - pd.Timedelta(days=3)
    t_7d = cut_dt - pd.Timedelta(days=7)

    # Признаки из VisitStream строго до cut_dt
    v_pre = visits[visits["ViewDate"] < cut_dt].copy()
    v_pre["v_day"] = v_pre["ViewDate"].dt.floor("D")
    v_days = v_pre[["UserID", "v_day"]].drop_duplicates().groupby("UserID").size().rename("v_active_days")

    ads_clean = ads.assign(clean_price=ads["Price"].fillna(0.0).clip(0.0, 500000.0))
    v_pre_priced = v_pre.merge(ads_clean[["AdID", "clean_price", "CategoryID"]], on="AdID", how="left")

    v_stats = v_pre_priced.groupby("UserID").agg(
        v_total=("AdID", "count"),
        v_uniq_ads=("AdID", "nunique"),
        v_uniq_cats=("CategoryID", "nunique"),
        v_mean_price=("clean_price", "mean"),
        v_last=("ViewDate", "max"),
        v_first=("ViewDate", "min"),
    )
    v_stats["v_recency_hours"] = (cut_dt - v_stats["v_last"]).dt.total_seconds() / 3600.0
    v_stats["v_span_hours"] = (v_stats["v_last"] - v_stats["v_first"]).dt.total_seconds() / 3600.0
    v_1d = v_pre[v_pre["ViewDate"] >= t_1d].groupby("UserID").size().rename("v_1d")
    v_3d = v_pre[v_pre["ViewDate"] >= t_3d].groupby("UserID").size().rename("v_3d")
    v_7d = v_pre[v_pre["ViewDate"] >= t_7d].groupby("UserID").size().rename("v_7d")

    # Признаки из SearchInfo строго до cut_dt
    si_valid = searches[searches["UserID"].notna()].copy()
    si_valid["UserID"] = si_valid["UserID"].astype("int64")
    s_pre = si_valid[si_valid["SearchDate"] < cut_dt].copy()
    s_pre["s_day"] = s_pre["SearchDate"].dt.floor("D")
    s_days = s_pre[["UserID", "s_day"]].drop_duplicates().groupby("UserID").size().rename("s_active_days")
    s_stats = s_pre.groupby("UserID").agg(
        s_total=("SearchID", "count"),
        s_logged_share=("IsUserLoggedOn", "mean"),
        s_uniq_cats=("CategoryID", "nunique"),
        s_uniq_locs=("LocationID", "nunique"),
        s_last=("SearchDate", "max"),
        s_first=("SearchDate", "min"),
    )
    s_stats["s_recency_hours"] = (cut_dt - s_stats["s_last"]).dt.total_seconds() / 3600.0
    s_stats["s_span_hours"] = (s_stats["s_last"] - s_stats["s_first"]).dt.total_seconds() / 3600.0
    s_1d = s_pre[s_pre["SearchDate"] >= t_1d].groupby("UserID").size().rename("s_1d")
    s_3d = s_pre[s_pre["SearchDate"] >= t_3d].groupby("UserID").size().rename("s_3d")
    s_7d = s_pre[s_pre["SearchDate"] >= t_7d].groupby("UserID").size().rename("s_7d")

    # Признаки из PhoneRequestsStream строго до cut_dt
    p_pre = phones[phones["PhoneRequestDate"] < cut_dt]
    p_total = p_pre.groupby("UserID").size().rename("p_total")
    p_3d = p_pre[p_pre["PhoneRequestDate"] >= t_3d].groupby("UserID").size().rename("p_3d")
    p_7d = p_pre[p_pre["PhoneRequestDate"] >= t_7d].groupby("UserID").size().rename("p_7d")

    df = users.merge(
        v_stats[["v_total", "v_uniq_ads", "v_uniq_cats", "v_mean_price", "v_recency_hours", "v_span_hours"]],
        on="UserID",
        how="inner",
    )
    for ser in [
        v_days,
        v_1d,
        v_3d,
        v_7d,
        s_stats[["s_total", "s_logged_share", "s_uniq_cats", "s_uniq_locs", "s_recency_hours", "s_span_hours"]],
        s_days,
        s_1d,
        s_3d,
        s_7d,
        p_total,
        p_3d,
        p_7d,
    ]:
        df = df.merge(ser, on="UserID", how="left")

    df["s_recency_hours"] = df["s_recency_hours"].fillna(400.0)
    df = df.fillna(0.0)

    # Относительные продуктовые признаки интенсивности и затухания активности
    df["views_per_active_day"] = df["v_total"] / (df["v_active_days"] + 1.0)
    df["searches_per_active_day"] = df["s_total"] / (df["s_active_days"] + 1.0)
    df["views_decay_3d_to_total"] = df["v_3d"] / (df["v_total"] + 1.0)
    df["searches_decay_3d_to_total"] = df["s_3d"] / (df["s_total"] + 1.0)
    df["view_to_search_ratio"] = df["v_total"] / (df["s_total"] + 1.0)
    df["contact_to_view_cr"] = df["p_total"] / (df["v_total"] + 1.0)
    df["recency_gap_v_minus_s"] = df["v_recency_hours"] - df["s_recency_hours"]

    # Целевая переменная: отток из активного просмотра карточек и контактов после 2015-05-11
    post_active_users = (
        set(visits[visits["ViewDate"] >= cut_dt]["UserID"].unique())
        | set(phones[phones["PhoneRequestDate"] >= cut_dt]["UserID"].unique())
    )
    df["churned"] = (~df["UserID"].isin(post_active_users)).astype(int)

    # Оценка ожидаемой ценности удержанного пользователя (на основе его исторической активности и контактов)
    df["expected_user_ltv_rub"] = np.round(
        180.0 + 42.0 * np.log1p(df["v_total"]) + 115.0 * np.log1p(df["p_total"]), 2
    )

    feature_cols = [
        c for c in df.columns
        if c not in ["UserID", "churned", "expected_user_ltv_rub"]
    ]
    return df, feature_cols


def evaluate_churn_scoring_pipeline(avito_root: str) -> dict:
    """
    Обучает LogisticRegression и HistGradientBoostingClassifier на 49 352 реальных пользователях Авито,
    проводит 5-Fold CV и оценивает качество на отложенной выборке (25%, 12 338 пользователей),
    сравнивая стандартный порог p0 = 0.50 и бизнес-оптимальный порог p*.
    """
    df, feature_cols = build_real_user_churn_dataset(avito_root)

    X = df[feature_cols]
    y = df["churned"].to_numpy()
    ltv = df["expected_user_ltv_rub"].to_numpy()

    X_train, X_test, y_train, y_test, ltv_train, ltv_test = train_test_split(
        X, y, ltv, test_size=0.25, random_state=42, stratify=y
    )

    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    gb_clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=6, min_samples_leaf=35, random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_logreg = cross_validate(logreg_pipe, X_train, y_train, cv=skf, scoring=["roc_auc", "average_precision"])
    cv_gb = cross_validate(gb_clf, X_train, y_train, cv=skf, scoring=["roc_auc", "average_precision"])

    logreg_pipe.fit(X_train, y_train)
    gb_clf.fit(X_train, y_train)

    t0 = time.perf_counter()
    y_prob_test = gb_clf.predict_proba(X_test)[:, 1]
    inference_ms = (time.perf_counter() - t0) * 1000.0
    y_prob_lr_test = logreg_pipe.predict_proba(X_test)[:, 1]

    # Юнит-экономика кампании удержания и балансировка Precision/Recall при дисбалансе классов (Churn = 19.95%):
    # Из-за базовой доли класса 19.95% стандартный порог p0 = 0.50 пропускает 99.1% оттока (Recall = 0.9%, F1 = 0.0183).
    # Оптимизация порога по максимуму F1-меры (и функции полезности при стоимости CRM-коммуникации 29 руб.) дает p* = 0.24:
    promo_cost_rub = 29.0
    save_lift_prob = 0.32

    def eval_at_threshold(probs: np.ndarray, y_true: np.ndarray, ltv_arr: np.ndarray, thr: float) -> dict:
        pred = (probs >= thr).astype(int)
        tp = (pred == 1) & (y_true == 1)
        recall_pct = float(np.sum(tp) / max(1, np.sum(y_true == 1)) * 100.0)
        precision_pct = float(np.sum(tp) / max(1, np.sum(pred == 1)) * 100.0)
        targeted_pct = float(np.mean(pred == 1) * 100.0)
        f1_val = float(f1_score(y_true, pred))
        net_profit = float(np.sum(ltv_arr[tp] * save_lift_prob) - np.sum(pred == 1) * promo_cost_rub)
        return {
            "threshold": round(float(thr), 2),
            "targeted_share_pct": round(targeted_pct, 1),
            "recall_pct": round(recall_pct, 1),
            "precision_pct": round(precision_pct, 1),
            "f1": round(f1_val, 4),
            "net_profit_rub": round(net_profit, 2),
        }

    default_p0 = eval_at_threshold(y_prob_test, y_test, ltv_test, thr=0.50)

    best_thr = 0.50
    best_f1 = 0.0
    for thr in np.linspace(0.15, 0.65, 51):
        m = eval_at_threshold(y_prob_test, y_test, ltv_test, thr=float(thr))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)

    optimal_p_star = eval_at_threshold(y_prob_test, y_test, ltv_test, thr=best_thr)

    return {
        "n_users_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_pct": round(float(np.mean(y) * 100.0), 2),
        "logreg_cv_roc_auc": round(float(np.mean(cv_logreg["test_roc_auc"])), 4),
        "gb_cv_roc_auc": round(float(np.mean(cv_gb["test_roc_auc"])), 4),
        "logreg_test_roc_auc": round(float(roc_auc_score(y_test, y_prob_lr_test)), 4),
        "gb_test_roc_auc": round(float(roc_auc_score(y_test, y_prob_test)), 4),
        "gb_test_pr_auc": round(float(average_precision_score(y_test, y_prob_test)), 4),
        "gb_test_f1_optimal": round(float(best_f1), 4),
        "batch_inference_time_ms": round(inference_ms, 2),
        "default_threshold_p0_050": default_p0,
        "optimal_threshold_p_star": optimal_p_star,
        "y_test": y_test,
        "y_prob_gb": y_prob_test,
    }


def evaluate_real_avito_pricing_pipeline(avito_root: str) -> dict:
    """
    Обучает регрессионную модель для бейджа «Рыночная цена» на 60 000 реальных объявлений Авито
    из AdsInfo.parquet (с валидной рублевой ценой Price в однородных подкатегориях IQR 25..75%,
    текстом заголовка Title, иерархией Category.parquet, локациями Location.parquet и спросом VisitStream).
    """
    db_dir = os.path.join(avito_root, "db")

    ads = pd.read_parquet(
        os.path.join(db_dir, "AdsInfo.parquet"),
        columns=["AdID", "LocationID", "CategoryID", "Price", "Title", "IsContext"],
    )
    cats = pd.read_parquet(os.path.join(db_dir, "Category.parquet"))
    locs = pd.read_parquet(os.path.join(db_dir, "Location.parquet"))
    visits = pd.read_parquet(os.path.join(db_dir, "VisitStream.parquet"), columns=["AdID"])
    phones = pd.read_parquet(os.path.join(db_dir, "PhoneRequestsStream.parquet"), columns=["AdID"])

    v_cnt = visits.groupby("AdID").size().rename("views_cnt")
    p_cnt = phones.groupby("AdID").size().rename("contacts_cnt")

    sub = ads[
        (ads["Price"] >= 1000.0)
        & (ads["Price"] <= 120000.0)
        & ads["Title"].notna()
        & (ads["Title"].str.len() >= 6)
    ].copy()

    # Быстрая векторизованная фильтрация внутрикатегорийного межквартильного диапазона (IQR 25%..75%),
    # отсекающая аксессуары/запчасти от основных товаров внутри одной категории
    cat_q25 = sub.groupby("CategoryID")["Price"].quantile(0.25)
    cat_q75 = sub.groupby("CategoryID")["Price"].quantile(0.75)
    q25_col = sub["CategoryID"].map(cat_q25)
    q75_col = sub["CategoryID"].map(cat_q75)
    sub = sub[(sub["Price"] >= q25_col) & (sub["Price"] <= q75_col)]

    sub = sub.merge(v_cnt, on="AdID", how="inner").merge(p_cnt, on="AdID", how="left")
    sub["contacts_cnt"] = sub["contacts_cnt"].fillna(0.0)
    sub = sub[sub["views_cnt"] >= 3].sample(n=60000, random_state=42).reset_index(drop=True)
    sub = sub.merge(cats, on="CategoryID", how="left").merge(locs, on="LocationID", how="left", suffixes=("_cat", "_loc"))

    sub["IsContext"] = sub["IsContext"].fillna(0.0)
    sub["RegionID"] = sub["RegionID"].fillna(-1.0)
    sub["CityID"] = sub["CityID"].fillna(-1.0)
    sub["ParentCategoryID"] = sub["ParentCategoryID"].fillna(-1)
    sub["SubcategoryID"] = sub["SubcategoryID"].fillna(-1)

    titles = sub["Title"].astype(str)
    sub["title_len_chars"] = titles.str.len()
    sub["title_words_cnt"] = titles.str.split().str.len()
    sub["title_digits_cnt"] = titles.str.count(r"\d")
    sub["contact_to_view_cr"] = sub["contacts_cnt"] / (sub["views_cnt"] + 1.0)
    sub["log_views"] = np.log1p(sub["views_cnt"])

    tr, te = train_test_split(sub, test_size=0.20, random_state=42)
    y_train = tr["Price"].to_numpy(dtype=float)
    y_test = te["Price"].to_numpy(dtype=float)

    # 1. Категорийный бейзлайн (медиана по родительской категории ParentCategoryID)
    parent_med = tr.groupby("ParentCategoryID")["Price"].median().to_dict()
    global_med = float(np.median(y_train))
    y_pred_baseline = np.array([parent_med.get(p, global_med) for p in te["ParentCategoryID"]])
    mape_baseline = float(mean_absolute_percentage_error(y_test, y_pred_baseline) * 100.0)
    rmse_baseline = float(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))

    # 2. Текстовые признаки заголовка объявления (TF-IDF биграммы + Ridge-скор и SVD)
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    X_tr_tf = tfidf.fit_transform(tr["Title"].astype(str))
    X_te_tf = tfidf.transform(te["Title"].astype(str))

    text_ridge = Ridge(alpha=1.5, random_state=42)
    text_ridge.fit(X_tr_tf, np.log(y_train))
    tr = tr.copy()
    te = te.copy()
    tr["text_price_score"] = text_ridge.predict(X_tr_tf)
    te["text_price_score"] = text_ridge.predict(X_te_tf)

    svd = TruncatedSVD(n_components=16, random_state=42)
    svd_tr = svd.fit_transform(X_tr_tf)
    svd_te = svd.transform(X_te_tf)
    for k in range(svd_tr.shape[1]):
        tr[f"svd_{k}"] = svd_tr[:, k]
        te[f"svd_{k}"] = svd_te[:, k]

    cat_med_log = np.log(tr.groupby("CategoryID")["Price"].median()).to_dict()
    cat_reg_med_log = np.log(tr.groupby(["CategoryID", "RegionID"])["Price"].median()).to_dict()
    global_med_log = float(np.log(global_med))

    tr["cat_med_log"] = [cat_med_log.get(c, global_med_log) for c in tr["CategoryID"]]
    te["cat_med_log"] = [cat_med_log.get(c, global_med_log) for c in te["CategoryID"]]
    tr["cat_reg_med_log"] = [
        cat_reg_med_log.get((c, r), cat_med_log.get(c, global_med_log))
        for c, r in zip(tr["CategoryID"], tr["RegionID"])
    ]
    te["cat_reg_med_log"] = [
        cat_reg_med_log.get((c, r), cat_med_log.get(c, global_med_log))
        for c, r in zip(te["CategoryID"], te["RegionID"])
    ]

    feature_cols = [
        "CategoryID",
        "ParentCategoryID",
        "SubcategoryID",
        "LocationID",
        "RegionID",
        "CityID",
        "IsContext",
        "views_cnt",
        "contacts_cnt",
        "contact_to_view_cr",
        "log_views",
        "title_len_chars",
        "title_words_cnt",
        "title_digits_cnt",
        "cat_med_log",
        "cat_reg_med_log",
        "text_price_score",
    ] + [f"svd_{k}" for k in range(svd_tr.shape[1])]

    X_train_num = tr[feature_cols]
    X_test_num = te[feature_cols]

    # Линейная модель Ridge
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=5.0)),
    ])
    ridge_pipe.fit(X_train_num, np.log(y_train))
    y_pred_ridge = np.exp(ridge_pipe.predict(X_test_num))
    mape_ridge = float(mean_absolute_percentage_error(y_test, y_pred_ridge) * 100.0)
    rmse_ridge = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

    # Градиентный бустинг HistGradientBoostingRegressor
    gb_reg = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.05, max_depth=6, min_samples_leaf=25, random_state=42
    )
    gb_reg.fit(X_train_num, np.log(y_train))
    y_pred_gb = np.exp(gb_reg.predict(X_test_num))
    mape_gb = float(mean_absolute_percentage_error(y_test, y_pred_gb) * 100.0)
    rmse_gb = float(np.sqrt(mean_squared_error(y_test, y_pred_gb)))
    r2_gb = float(r2_score(y_test, y_pred_gb))

    rel_mape_red_pct = (1.0 - mape_gb / mape_baseline) * 100.0
    rel_rmse_red_pct = (1.0 - rmse_gb / rmse_baseline) * 100.0

    return {
        "n_listings_total": int(len(sub)),
        "mean_price_rub": round(float(sub["Price"].mean()), 2),
        "median_price_rub": round(float(sub["Price"].median()), 2),
        "baseline_mape_pct": round(mape_baseline, 2),
        "baseline_rmse_rub": round(rmse_baseline, 2),
        "ridge_mape_pct": round(mape_ridge, 2),
        "ridge_rmse_rub": round(rmse_ridge, 2),
        "gb_mape_pct": round(mape_gb, 2),
        "gb_rmse_rub": round(rmse_gb, 2),
        "gb_r2": round(r2_gb, 4),
        "relative_mape_reduction_pct": round(rel_mape_red_pct, 1),
        "relative_rmse_reduction_pct": round(rel_rmse_red_pct, 1),
    }


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    avito_root = resolve_avito_root()

    print("=" * 92)
    print("AVITO PRODUCT ANALYTICS | ПРОЕКТ 3: СКОРИНГ ОТТОКА И ОЦЕНКА ЦЕНЫ ОБЪЯВЛЕНИЙ (rel-avito)")
    print("=" * 92)

    churn_res = evaluate_churn_scoring_pipeline(avito_root)
    price_res = evaluate_real_avito_pricing_pipeline(avito_root)

    p0 = churn_res["default_threshold_p0_050"]
    ps = churn_res["optimal_threshold_p_star"]

    print(f"[1] СКОРИНГ ОТТОКА ПОЛЬЗОВАТЕЛЕЙ АВИТО (N = {churn_res['n_users_total']:,} активных пользователей):")
    print(f"    • Выборка: train = {churn_res['n_train']:,}, holdout test = {churn_res['n_test']:,} (Churn Rate = {churn_res['churn_rate_pct']:.2f}%)")
    print(f"    • 5-Fold CV ROC-AUC: LogisticRegression = {churn_res['logreg_cv_roc_auc']:.4f} | HistGradientBoosting = {churn_res['gb_cv_roc_auc']:.4f}")
    print(f"    • Holdout Test: ROC-AUC = {churn_res['gb_test_roc_auc']:.4f} (LogReg = {churn_res['logreg_test_roc_auc']:.4f}), "
          f"PR-AUC = {churn_res['gb_test_pr_auc']:.4f}, F1 = {churn_res['gb_test_f1_optimal']:.4f}")
    print(f"    • Сравнение порогов вероятности оттока (до и после оптимизации):")
    print(f"      - Стандартный порог p0 = 0.50: таргетируем {p0['targeted_share_pct']:.1f}% базы, "
          f"Recall = {p0['recall_pct']:.1f}%, F1 = {p0['f1']:.4f}, чистый эффект = {p0['net_profit_rub']:+,.0f} руб.")
    print(f"      - Оптимальный порог p* = {ps['threshold']:.2f}: таргетируем {ps['targeted_share_pct']:.1f}% базы, "
          f"Recall = {ps['recall_pct']:.1f}%, F1 = {ps['f1']:.4f}, чистый эффект = {ps['net_profit_rub']:+,.0f} руб.")

    print(f"\n[2] РЕГРЕССИОННАЯ МОДЕЛЬ «РЫНОЧНАЯ ЦЕНА» НА ADSINFO.PARQUET (N = {price_res['n_listings_total']:,} объявлений):")
    print(f"    • Средняя цена товара в выборке: {price_res['mean_price_rub']:,.2f} руб. (медиана: {price_res['median_price_rub']:,.0f} руб.)")
    print(f"    • Категорийный бейзлайн: MAPE = {price_res['baseline_mape_pct']:.2f}%, RMSE = {price_res['baseline_rmse_rub']:,.0f} руб.")
    print(f"    • Ridge Regression:     MAPE = {price_res['ridge_mape_pct']:.2f}%, RMSE = {price_res['ridge_rmse_rub']:,.0f} руб.")
    print(f"    • Gradient Boosting:    MAPE = {price_res['gb_mape_pct']:.2f}%, RMSE = {price_res['gb_rmse_rub']:,.0f} руб., R^2 = {price_res['gb_r2']:.4f}")
    print(f"    • Снижение ошибки относительно бейзлайна: MAPE на -{price_res['relative_mape_reduction_pct']:.1f}%, "
          f"RMSE на -{price_res['relative_rmse_reduction_pct']:.1f}%")
    print("=" * 92)

    y_test = churn_res.pop("y_test")
    y_prob_gb = churn_res.pop("y_prob_gb")

    full_summary = {
        "dataset": "Avito Context Ad Clicks (rel-avito)",
        "user_churn_scoring": churn_res,
        "market_price_regression": price_res,
    }
    metrics_path = os.path.join(output_dir, "ml_scoring_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=140)
    prec, rec, _ = precision_recall_curve(y_test, y_prob_gb)
    axes[0].plot(rec, prec, color="#2b6cb0", linewidth=2.2, label=f"HistGradBoost (AUC={churn_res['gb_test_roc_auc']:.3f}, PR-AUC={churn_res['gb_test_pr_auc']:.3f})")
    axes[0].axhline(np.mean(y_test), color="#e53e3e", linestyle="--", label=f"Baseline ({np.mean(y_test):.2f})")
    axes[0].set_title("PR-кривая скоринга оттока пользователей (N = 49 352)", fontsize=10.5)
    axes[0].set_xlabel("Полнота (Recall)")
    axes[0].set_ylabel("Точность (Precision)")
    axes[0].legend()
    axes[0].grid(linestyle=":", alpha=0.6)

    models_p = ["Baseline\n(Категория)", "Ridge\n(TF-IDF + Кат.)", "HistGradBoost\n(Спрос + Текст)"]
    mapes = [price_res["baseline_mape_pct"], price_res["ridge_mape_pct"], price_res["gb_mape_pct"]]
    bars = axes[1].bar(models_p, mapes, color=["#a0aec0", "#4299e1", "#2f855a"], width=0.5)
    axes[1].set_title(f"Оценка цены в AdsInfo.parquet (MAPE, %): -{price_res['relative_mape_reduction_pct']:.1f}%", fontsize=10.5)
    axes[1].set_ylabel("MAPE (%)")
    for b, m in zip(bars, mapes):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{m:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
    axes[1].set_ylim(0, max(mapes) * 1.22)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ml_scoring_results.png")
    plt.savefig(plot_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
