"""
train_model.py — Random Forest على داتاسيت Q-CHAT-10 الحقيقي
شغّل مرة واحدة: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# ══════════════════════════════════════════════════════
# 1. تحميل الداتاسيت
# ══════════════════════════════════════════════════════
df = pd.read_csv('Toddler_Autism_dataset_July_2018__1_.csv')

print("=" * 55)
print("         تحليل الداتاسيت")
print("=" * 55)
print(f"  عدد الصفوف:  {len(df)}")
print(f"  Not at Risk: {(df['Class/ASD Traits '] == 'No').sum()}")
print(f"  At Risk:     {(df['Class/ASD Traits '] == 'Yes').sum()}")

# ══════════════════════════════════════════════════════
# 2. إعداد الـ Features
# ══════════════════════════════════════════════════════
# الأسئلة العشرة: 0 أو 1 (من الداتاسيت مباشرة)
# عند استخدام الـ UI:
#   Always   → 1
#   Sometimes → 1
#   Never    → 0
q_cols = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']

# تحويل المتغيرات النصية
df['Sex_enc']      = (df['Sex'] == 'm').astype(int)
df['Jaundice_enc'] = (df['Jaundice'] == 'yes').astype(int)
df['Family_enc']   = (df['Family_mem_with_ASD'] == 'yes').astype(int)

feature_cols = q_cols + ['Age_Mons', 'Sex_enc', 'Jaundice_enc', 'Family_enc']

X = df[feature_cols].values
y = (df['Class/ASD Traits '] == 'Yes').astype(int).values

# ══════════════════════════════════════════════════════
# 3. تقسيم البيانات 80/20
# ══════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

# ══════════════════════════════════════════════════════
# 4. بناء Random Forest
# ══════════════════════════════════════════════════════
model = RandomForestClassifier(
    n_estimators=200,       # عدد الأشجار
    max_depth=10,           # عمق الشجرة
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced', # يعالج class imbalance (728 vs 326)
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ══════════════════════════════════════════════════════
# 5. التقييم الكامل
# ══════════════════════════════════════════════════════
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 55)
print("       نتائج التقييم — Random Forest")
print("=" * 55)
print(f"  Accuracy:   {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision:  {prec:.4f}  ({prec*100:.2f}%)")
print(f"  Recall:     {rec:.4f}  ({rec*100:.2f}%)")
print(f"  F1-Score:   {f1:.4f}  ({f1*100:.2f}%)")
print(f"  ROC-AUC:    {auc:.4f}  ({auc*100:.2f}%)")

# Cross-Validation (5-Fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"\n  CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  CV ROC-AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:")
print(f"                Predicted No  Predicted Yes")
print(f"  Actual No  :      {cm[0,0]:>4}          {cm[0,1]:>4}")
print(f"  Actual Yes :      {cm[1,0]:>4}          {cm[1,1]:>4}")

print(f"\n{classification_report(y_test, y_pred, target_names=['Not at Risk','At Risk'])}")

# Feature Importance
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("  Feature Importance:")
for name, val in fi.items():
    bar = "█" * int(val * 60)
    print(f"  {name:15s} {val:.4f}  {bar}")

# ══════════════════════════════════════════════════════
# 6. حفظ الموديل والـ metadata
# ══════════════════════════════════════════════════════
joblib.dump(model, 'rf_model.pkl')

meta = {
    'feature_cols': feature_cols,
    'q_cols': q_cols,
    'encode_map': {
        'Always': 1, 'Sometimes': 1, 'Never': 0
    },
    'metrics': {
        'accuracy':  round(acc, 4),
        'precision': round(prec, 4),
        'recall':    round(rec, 4),
        'f1':        round(f1, 4),
        'roc_auc':   round(auc, 4),
    },
    'feature_importance': fi.round(4).to_dict()
}

with open('model_meta.json', 'w', ensure_ascii=False) as f:
    json.dump(meta, f, indent=2)

print("\n✓ تم حفظ الموديل:    rf_model.pkl")
print("✓ تم حفظ المعلومات: model_meta.json")
print("=" * 55)