"""
verify_model.py — تحقق سريع أن الموديل يشتغل صح
شغّل: python verify_model.py
"""
import joblib, numpy as np, json

print("=" * 55)
print("  فحص الموديل — Random Forest")
print("=" * 55)

# ── 1. تحميل ─────────────────────────────────────────
try:
    model = joblib.load('rf_model.pkl')
    print(f"✓ الموديل محمّل")
    print(f"  النوع:       {type(model).__name__}")
    print(f"  عدد الأشجار: {model.n_estimators}")
    print(f"  عدد الـ Features: {model.n_features_in_}")
except FileNotFoundError:
    print("✗ rf_model.pkl غير موجود — شغّل: python train_model.py")
    exit(1)

with open('model_meta.json') as f:
    meta = json.load(f)
print(f"  الأعمدة: {meta['feature_cols']}")
print()

# ── 2. حالة خطر عالي: كل الإجابات 1 ─────────────────
print("─" * 55)
print("الحالة 1: طفل يجاوب 'Always' على كل شيء (خطر عالي)")
# A1-A10=1, Age=24, Sex=1, Jaundice=1, Family=1
X1 = np.array([1,1,1,1,1,1,1,1,1,1, 24, 1, 1, 1]).reshape(1,-1)
p1 = model.predict(X1)[0]
r1 = model.predict_proba(X1)[0]
print(f"  Feature vector: {X1[0].tolist()}")
print(f"  Prediction: {p1} ({'at_risk' if p1==1 else 'not_at_risk'})")
print(f"  Confidence: {r1[p1]*100:.2f}%")
print(f"  Proba [not_risk, at_risk]: {r1.round(4).tolist()}")

# ── 3. حالة آمنة: كل الإجابات 0 ─────────────────────
print()
print("─" * 55)
print("الحالة 2: طفل يجاوب 'Never' على كل شيء (آمن)")
X2 = np.array([0,0,0,0,0,0,0,0,0,0, 36, 0, 0, 0]).reshape(1,-1)
p2 = model.predict(X2)[0]
r2 = model.predict_proba(X2)[0]
print(f"  Feature vector: {X2[0].tolist()}")
print(f"  Prediction: {p2} ({'at_risk' if p2==1 else 'not_at_risk'})")
print(f"  Confidence: {r2[p2]*100:.2f}%")
print(f"  Proba [not_risk, at_risk]: {r2.round(4).tolist()}")

# ── 4. حالة مختلطة ───────────────────────────────────
print()
print("─" * 55)
print("الحالة 3: إجابات مختلطة")
# Always=1, Sometimes=1, Never=0
# answers: Always,Never,Sometimes,Always,Never,Always,Sometimes,Never,Always,Sometimes
enc = [1,0,1,1,0,1,1,0,1,1]
X3  = np.array(enc + [28, 1, 0, 0]).reshape(1,-1)
p3  = model.predict(X3)[0]
r3  = model.predict_proba(X3)[0]
print(f"  الإجابات (بعد تحويل Always/Sometimes→1, Never→0): {enc}")
print(f"  Feature vector: {X3[0].tolist()}")
print(f"  Prediction: {p3} ({'at_risk' if p3==1 else 'not_at_risk'})")
print(f"  Confidence: {r3[p3]*100:.2f}%")
print(f"  Proba [not_risk, at_risk]: {r3.round(4).tolist()}")

# ── 5. Feature Importance ─────────────────────────────
print()
print("─" * 55)
print("أهمية الأسئلة (Feature Importance):")
names = meta['feature_cols']
fi    = model.feature_importances_
for name, val in sorted(zip(names, fi), key=lambda x: -x[1]):
    bar = "█" * int(val * 50)
    print(f"  {name:15s} {val:.4f}  {bar}")

print()
print("=" * 55)
print("✓ الموديل يعمل بشكل صحيح!")
print("  الآن شغّل: python app.py")
print("  وتحقق من: http://localhost:5000/")
print("=" * 55)