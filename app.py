"""
app.py — Flask Backend لنظام فحص التوحد
تشغيل: python app.py
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client
import os
from dotenv import load_dotenv
import joblib, numpy as np, json, logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
# ══════════════════════════════════════════════════════
# تحميل الموديل
# ══════════════════════════════════════════════════════
try:
    model = joblib.load('rf_model.pkl')
    log.info("✓ الموديل محمّل: rf_model.pkl | نوعه: %s", type(model).__name__)
    log.info("  عدد الأشجار: %d | Features: %d", model.n_estimators, model.n_features_in_)
except FileNotFoundError:
    log.error("✗ الملف rf_model.pkl غير موجود — شغّل train_model.py أولاً")
    raise SystemExit(1)

with open('model_meta.json', encoding='utf-8') as f:
    META = json.load(f)
log.info("✓ Model metrics: %s", META['metrics'])

# ── Supabase ──────────────────────────────────────────
try:
    supabase = create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_ANON_KEY']
    )
    log.info("✓ Supabase متصل")
except KeyError as e:
    log.warning("⚠ متغير البيئة %s غير موجود — الحفظ لن يعمل", e)
    supabase = None

# ── Encoding ──────────────────────────────────────────
ENCODE = {'Always': 1, 'Sometimes': 1, 'Never': 0}

# A1-A9: Never أو Sometimes = 1 (خطر)، Always = 0
# A10:   Always أو Sometimes = 1 (خطر)، Never = 0
def encode_answers(raw: list) -> list:
    result = []
    for i, a in enumerate(raw):
        if isinstance(a, str):
            if a not in ('Always', 'Sometimes', 'Never'):
                raise ValueError(f"إجابة غير معروفة: '{a}'")
            if i == 9:  # A10 معكوسة
                result.append(1 if a in ('Always', 'Sometimes') else 0)
            else:
                result.append(1 if a in ('Never', 'Sometimes') else 0)
        else:
            v = int(a)
            if v not in (0, 1):
                raise ValueError(f"قيمة غير صحيحة: {v}")
            result.append(v)
    return result

def get_risk_level(score: int) -> str:
    if score >= 7:   return 'high'
    elif score >= 4: return 'medium'
    else:            return 'low'

def get_or_create_user_id(user_name: str, user_email: str) -> str:
    """
    يجيب id المستخدم لو موجود، أو ينشئه — بدون ما يلمس password_hash الموجود.
    هذي الدالة هي الإصلاح الرئيسي لمشكلة تخريب كلمة المرور.
    """
    existing = supabase.table('users').select('id').eq('email', user_email).execute()
    if existing.data:
        # المستخدم موجود — نرجع id فقط بدون أي تعديل
        return existing.data[0]['id']
    else:
        # مستخدم جديد — ننشئه بكلمة مرور مؤقتة
        # (المستخدم المفروض يسجل من صفحة التسجيل ويغير كلمة المرور)
        new_user = supabase.table('users').insert({
            'name':          user_name,
            'email':         user_email,
            'password_hash': generate_password_hash('temp_' + user_email),
            'role':          'parent'
        }).execute()
        return new_user.data[0]['id']


# ══════════════════════════════════════════════════════
# ROUTE  — عرض التقرير 
# ════════════════════════════════════════════

@app.route('/api/screening/<screening_id>')
def get_screening(screening_id):
    if not supabase:
        return jsonify({'error': 'Database not connected'}), 503

    try:
        res = supabase.table('screenings').select('*').eq('id', screening_id).execute()

        if not res.data:
            return jsonify({'error': 'Screening not found'}), 404

        screening = res.data[0]

        screening['answers'] = [
        screening.get(f'q{i}') or ''
        for i in range(1, 11)
    ]

        pred_res = supabase.table('predictions') \
            .select('*') \
            .eq('screening_id', screening_id) \
            .execute()

        prediction = pred_res.data[0] if pred_res.data else {}

        return jsonify({**screening, 'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500   
     
# ══════════════════════════════════════════════════════
# ROUTE 1 — فحص الخادم + معلومات الموديل
# ══════════════════════════════════════════════════════
@app.route('/api/health')
def health():
    return jsonify({
        'status':   'Autism Screening API ✓',
        'model':    type(model).__name__,
        'trees':    model.n_estimators,
        'features': model.n_features_in_,
        'metrics':  META['metrics'],
        'supabase': supabase is not None
    })


# ══════════════════════════════════════════════════════
# ROUTE 2 — اختبار الموديل مباشرة بدون DB
# GET /api/test-model?answers=1,0,1,1,0,1,1,0,1,1&age=28&sex=1&jaundice=0&family=0
# ══════════════════════════════════════════════════════
@app.route('/api/test-model')
def test_model():
    try:
        raw = request.args.get('answers', '1,0,1,1,0,1,1,0,1,1')
        age = int(request.args.get('age',      28))
        sex = int(request.args.get('sex',       1))
        jau = int(request.args.get('jaundice',  0))
        fam = int(request.args.get('family',    0))

        answers = [int(x) for x in raw.split(',')]
        if len(answers) != 10:
            return jsonify({'error': 'يجب 10 إجابات'}), 400

        X    = np.array(answers + [age, sex, jau, fam]).reshape(1, -1)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        result = 'at_risk' if pred == 1 else 'not_at_risk'
        conf   = float(prob[pred])

        log.info("[TEST-MODEL] answers=%s → result=%s conf=%.4f", answers, result, conf)

        return jsonify({
            'source':         'random_forest_model',
            'model_type':     type(model).__name__,
            'n_trees':        model.n_estimators,
            'input_features': (answers + [age, sex, jau, fam]),
            'feature_names':  META['feature_cols'],
            'raw_proba':      {'not_at_risk': round(float(prob[0]), 4), 'at_risk': round(float(prob[1]), 4)},
            'prediction':     int(pred),
            'result':         result,
            'confidence':     round(conf, 4),
            'total_score':    sum(answers),
            'risk_level':     get_risk_level(sum(answers))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# @app.route('/api/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.get_json()
    if not body:
        return jsonify({'error': 'الطلب فارغ'}), 400

    required = ['user_name', 'user_email', 'child_name',
                'child_age_months', 'child_gender', 'answers',
                'jaundice', 'family_asd']

    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({'error': f'حقول مفقودة: {missing}'}), 400

    answers_raw = body['answers']
    if len(answers_raw) != 10:
        return jsonify({'error': 'يجب 10 إجابات بالضبط'}), 400

    try:
        answers = encode_answers(answers_raw)
    except ValueError as e:
        return jsonify({'error': str(e)}), 422

    q_score = sum(answers)
    sex_enc      = 1 if body.get('child_gender') == 'male' else 0
    jaundice_enc = 1 if body.get('jaundice') == 'yes' else 0
    family_enc   = 1 if body.get('family_asd') == 'yes' else 0
    age_mons     = int(body.get('child_age_months', 24))

    feature_vector = answers + [age_mons, sex_enc, jaundice_enc, family_enc]
    X = np.array(feature_vector).reshape(1, -1)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    result = 'at_risk' if pred == 1 else 'not_at_risk'
    confidence = round(float(prob[pred]), 4)

    if float(prob[1]) >= 0.70:
        risk_level = 'high'
    elif float(prob[1]) >= 0.40:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    log.info("Child: %s | Age: %d", body['child_name'], age_mons)

    sc_id = None
    db_saved = False

    try:
        if supabase:
            user_id = get_or_create_user_id(body['user_name'], body['user_email'])

            q_vals = {f'q{i+1}': answers[i] for i in range(10)}

            sc = supabase.table('screenings').insert({
                'user_id': user_id,
                'child_name': body['child_name'],
                'child_age_months': age_mons,
                'child_gender': body['child_gender'],
                'jaundice': body.get('jaundice', 'no'),
                'family_asd': body.get('family_asd', 'no'),
                'total_score': q_score,
                'acceptance_status': 'pending',
                'assigned_doctor_id': body.get('doctor_id'),
                **q_vals
            }).execute()

            if sc.data:
                sc_id = sc.data[0].get('id')

            supabase.table('predictions').insert({
                'screening_id': sc_id,
                'result': result,
                'risk_level': risk_level,
                'confidence': confidence,
            }).execute()

            db_saved = True

    except Exception as e:
        log.error(e)
        db_saved = False

    risk_prob = float(prob[1]) if prob is not None and len(prob) > 1 else 0.0

    return jsonify({
        'source': 'random_forest_model',
        'model_type': type(model).__name__,
        'screening_id': sc_id,
        'db_saved': db_saved,
        'result': result,
        'risk_level': risk_level,
        'confidence': confidence,
        'at_risk_probability': round(risk_prob, 4),
        'total_score': q_score,
        'max_score': 10,
        'feature_vector': feature_vector
    })
# ══════════════════════════════════════════════════════
# ROUTE — التسجيل
# ══════════════════════════════════════════════════════
@app.route('/api/register', methods=['POST'])
@app.route('/api/register/', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'الطلب فارغ'}), 400

    name      =  data.get('name', '').strip()
    email     = data.get('email', '').strip().lower()
    password  = data.get('password', '')
    role      = data.get('role', '')
    specialty = data.get('specialty', None)

    # ── تحقق من الحقول
    if not name or not email or not password or not role:
        return jsonify({'error': 'Missing required fields'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    if role not in ('parent', 'specialist', 'admin'):
        return jsonify({'error': 'Invalid role'}), 400

    if not supabase:
        return jsonify({'error': 'Database not connected'}), 503

    # ── تحقق من البريد المكرر
    existing = supabase.table('users').select('id').eq('email', email).execute()
    if existing.data:
        return jsonify({'error': 'Email already exists'}), 400

    # ── إنشاء المستخدم
    hashed_password = generate_password_hash(password)
    new_user = {
        'name':          name,
        'email':         email,
        'password_hash': hashed_password,
        'role':          role,
        'specialty':     specialty if role == 'specialist' else None
    }
    result = supabase.table('users').insert(new_user).execute()

    if not result.data:
        return jsonify({'error': 'Failed to create user'}), 500

    log.info("[REGISTER] ✓ مستخدم جديد: %s | role: %s", email, role)
    return jsonify({'message': 'User created successfully'}), 201


# ══════════════════════════════════════════════════════
# ROUTE — تسجيل الدخول
# ══════════════════════════════════════════════════════
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'wrong'}), 401

    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')
    role     = data.get('role', '')

    if not email or not password:
        return jsonify({'error': 'wrong'}), 401

    if not supabase:
        return jsonify({'error': 'wrong'}), 503

    # ── البحث عن المستخدم
    res = supabase.table('users').select('*').eq('email', email).execute()
    if not res.data:
        log.warning("[LOGIN] ✗ بريد غير موجود: %s", email)
        return jsonify({'error': 'wrong'}), 401

    user = res.data[0]

    # ── التحقق من كلمة المرور
    if not check_password_hash(user['password_hash'], password):
        log.warning("[LOGIN] ✗ كلمة مرور خاطئة: %s", email)
        return jsonify({'error': 'wrong'}), 401

    # ── التحقق من الدور
    if user['role'] != role:
        log.warning("[LOGIN] ✗ دور خاطئ: %s (طلب %s لكن هو %s)", email, role, user['role'])
        return jsonify({'error': 'role_mismatch'}), 401

    # ── تحديد صفحة التوجيه
    redirect_map = {
        'admin':      '/admin',
        'specialist': '/specialist',
        'parent':     '/parent'
    }
    redirect_page = redirect_map.get(user['role'], '/parent')

    log.info("[LOGIN] ✓ دخول ناجح: %s | role: %s", email, user['role'])

    return jsonify({
        'email':    user['email'],
        'name':     user['name'],
        'role':     user['role'],
        'specialty': user.get('specialty'),
        'redirect': redirect_page
    }), 200


# ══════════════════════════════════════════════════════
# ROUTE — تعيين طبيب
# ══════════════════════════════════════════════════════
@app.route('/api/assign-doctor', methods=['POST'])
def assign_doctor():
    body   = request.get_json()
    sc_id  = body.get('screening_id')
    doc_id = body.get('doctor_id')
    if not sc_id or not doc_id:
        return jsonify({'error': 'screening_id و doctor_id مطلوبان'}), 400
    if supabase:
        try:
            supabase.table('screenings').update(
                {'assigned_doctor_id': doc_id}
            ).eq('id', sc_id).execute()
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'success': True})


# ══════════════════════════════════════════════════════
# ROUTE — قبول الحالة من المختص
# ══════════════════════════════════════════════════════
@app.route('/api/accept-case', methods=['POST'])
def accept_case():
    body = request.get_json()
    sc_id = body.get('screening_id')
    if not sc_id:
        return jsonify({'error': 'screening_id مطلوب'}), 400
    if not supabase:
        return jsonify({'error': 'Database not connected'}), 503

    try:
        result = supabase.table('screenings').update(
            {'acceptance_status': 'accepted'}
        ).eq('id', sc_id).execute()
        
        if not result.data:
            log.warning("[ACCEPT-CASE] ✗ لم يتم العثور على الحالة: %s", sc_id)
            return jsonify({'error': 'Screening not found'}), 404

        log.info("[ACCEPT-CASE] ✓ تم قبول الحالة: %s", sc_id)
        return jsonify({'success': True})
    except Exception as e:
        log.error("[ACCEPT-CASE] ✗ خطأ في Supabase: %s", e)
        return jsonify({'error': str(e)}), 500

# ══════════════════════════════════════════════════════
# ROUTE — سجل ولي الأمر
# ══════════════════════════════════════════════════════
@app.route('/api/history/<path:email>')
def history(email):
    if not supabase:
        return jsonify([])
    try:
        u = supabase.table('users').select('id').eq('email', email).execute()
        if not u.data:
            return jsonify({'error': 'مستخدم غير موجود'}), 404
        rows = supabase.table('screenings') \
                 .select('*, predictions(*)') \
                 .eq('user_id', u.data[0]['id']) \
                 .order('created_at', desc=True).execute()
        return jsonify(rows.data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════
# ROUTE — إحصائيات الأدمن
# ══════════════════════════════════════════════════════
@app.route('/api/stats')
def stats():
    base = {
        'model_type':         type(model).__name__,
        'model_trees':        model.n_estimators,
        'model_metrics':      META['metrics'],
        'supabase_connected': supabase is not None
    }
    if not supabase:
        return jsonify({**base, 'total_screenings': 0, 'at_risk': 0, 'not_at_risk': 0})
    try:
        total = supabase.table('screenings').select('id', count='exact').execute()
        risk  = supabase.table('predictions').select('id', count='exact').eq('result', 'at_risk').execute()
        users = supabase.table('users').select('id', count='exact').eq('role', 'parent').execute()
        docs  = supabase.table('users').select('id', count='exact').eq('role', 'specialist').execute()
        return jsonify({
            **base,
            'total_screenings': total.count or 0,
            'at_risk':          risk.count  or 0,
            'not_at_risk':      (total.count or 0) - (risk.count or 0),
            'total_parents':    users.count or 0,
            'total_doctors':    docs.count  or 0,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════
# ROUTE — قائمة الأطباء
# ══════════════════════════════════════════════════════
@app.route('/api/doctors')
def get_doctors():
    if not supabase:
        return jsonify([])
    try:
        docs = supabase.table('users').select('id,name,email,specialty').eq('role', 'specialist').execute()
        return jsonify(docs.data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════
# ROUTE — الأطفال عند مختص معين
# ══════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════
# ROUTE — الأطفال عند مختص معين
# ══════════════════════════════════════════════════════
@app.route('/api/specialist/<doctor_id>/children')
def specialist_children(doctor_id):
    if not supabase:
        return jsonify([])
    try:
        # 1. جلب الفحوصات والتنبؤات
        rows = supabase.table('screenings') \
                 .select('*, predictions(*)') \
                 .eq('assigned_doctor_id', doctor_id) \
                 .order('created_at', desc=True).execute()
        
        data = rows.data
        
        # 2. استخراج معرفات أولياء الأمور (بدون تكرار)
        user_ids = list(set([r['user_id'] for r in data if r.get('user_id')]))
        
        # 3. جلب الأسماء والإيميلات من جدول المستخدمين ودمجها
        if user_ids:
            users_res = supabase.table('users').select('id, name, email').in_('id', user_ids).execute()
            user_map = {u['id']: u for u in users_res.data}
            
            for r in data:
                u = user_map.get(r.get('user_id'), {})
                r['user_name'] = u.get('name', '—')
                r['user_email'] = u.get('email', '—')

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ══════════════════════════════════════════════════════
# أضف هذين الـ routes لـ app.py (قبل if __name__ == '__main__':)
# ══════════════════════════════════════════════════════

# ── ROUTE: جلب كل المستخدمين (للأدمن فقط) ────────────
@app.route('/api/users')
def get_users():
    if not supabase:
        return jsonify({'error': 'Database not connected'}), 503
    try:
        res = supabase.table('users') \
                .select('id, name, email, role, specialty, created_at') \
                .order('created_at', desc=False).execute()
        return jsonify(res.data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ROUTE: حذف مستخدم (للأدمن فقط) ──────────────────
@app.route('/api/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    if not supabase:
        return jsonify({'error': 'Database not connected'}), 503
    try:
        # تأكد أنه مش admin قبل الحذف
        user = supabase.table('users').select('role').eq('id', user_id).execute()
        if not user.data:
            return jsonify({'error': 'User not found'}), 404
        if user.data[0]['role'] == 'admin':
            return jsonify({'error': 'Cannot delete admin accounts'}), 403

        supabase.table('users').delete().eq('id', user_id).execute()
        log.info("[DELETE USER] ✓ حُذف المستخدم: %s", user_id)
        return jsonify({'success': True})
    except Exception as e:
        log.error("[DELETE USER] ✗ %s", e)
        return jsonify({'error': str(e)}), 500


# ── ROUTE: كل الفحوصات (للأدمن) ──────────────────────
@app.route('/api/admin/screenings')
def admin_screenings():
    if not supabase:
        return jsonify({'error': 'Database not connected'}), 503
    try:
        # جلب الفحوصات مع predictions
        rows = supabase.table('screenings') \
                 .select('*, predictions(*)') \
                 .order('created_at', desc=True).execute()

        data = rows.data

        # جلب أسماء أولياء الأمور
        user_ids = list(set([r['user_id'] for r in data if r.get('user_id')]))
        if user_ids:
            users_res = supabase.table('users').select('id, name, email').in_('id', user_ids).execute()
            user_map  = {u['id']: u for u in users_res.data}
            for r in data:
                u = user_map.get(r.get('user_id'), {})
                r['user_name']  = u.get('name', '—')
                r['user_email'] = u.get('email', '—')

        # جلب أسماء الأطباء المعيّنين
        doc_ids = list(set([r['assigned_doctor_id'] for r in data if r.get('assigned_doctor_id')]))
        if doc_ids:
            docs_res = supabase.table('users').select('id, name').in_('id', doc_ids).execute()
            doc_map  = {d['id']: d['name'] for d in docs_res.data}
            for r in data:
                did = r.get('assigned_doctor_id')
                r['assigned_doctor_name'] = doc_map.get(did, '—') if did else '—'

        return jsonify(data)
    except Exception as e:
        log.error("[ADMIN SCREENINGS] ✗ %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/admin')
def admin_page():
    return render_template('Admin.html')

@app.route('/parent')
def parent_page():
    return render_template('Parent.html')

@app.route('/specialist')
def specialist_page():
    return render_template('Specialist.html')
