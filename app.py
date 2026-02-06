import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

# --- إعدادات الواجهة (Streamlit UI) ---
st.set_page_config(page_title="Car Price Expert", page_icon="🏎️")
st.title("🏎️ AI Car Price Predictor")
st.write("Welcome! This app uses Machine Learning to estimate car prices based on history.")
st.markdown("---") # خط فاصل للتنظيم

# --- 1. جلب البيانات (Data Loading) ---
url = "https://raw.githubusercontent.com/PhilopateerDev/Car-Price-Project./main/Car%20details.csv"
df_raw = pd.read_csv(url)

# عرض البيانات الأصلية عشان المستخدم يشوف إحنا شغالين على إيه
st.subheader("📊 Historical Data Preview")
st.write(df_raw.head())

# --- 2. معالجة البيانات (Preprocessing) ---

# استخراج أول كلمتين (الماركة والموديل) عشان الدقة تكون أعلى
# مثال: 'Maruti Swift VXI' بتبقى 'Maruti Swift'
df_raw['brand_model'] = df_raw['name'].str.split(' ').str.slice(0, 2).str.join(' ')

# هنعمل نسخة من البيانات عشان نشتغل عليها ونحولها لأرقام
df = df_raw.copy()

# قاموس سحري عشان نخزن فيه الـ Encoders ونستخدمها في تحويل مدخلات المستخدم لاحقاً
le_dict = {}
categorical_cols = ['brand_model', 'fuel', 'seller_type', 'transmission', 'owner']

# تحويل كل الأعمدة النصية لأرقام أوتوماتيكياً
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le # بنحفظ المترجم هنا عشان نحتاجه تحت

# تنظيف الجدول النهائي: حذف عمود الاسم القديم وتحديد الأهداف
df_final = df.drop(['name'], axis=1)
y = df_final['selling_price'] # الهدف (السعر)
X = df_final.drop(['selling_price'], axis=1) # المعطيات (كل شيء ما عدا السعر)

# --- 3. بناء وتدريب الموديل (Machine Learning) ---

# تقسيم البيانات: 80% للتدريب و 20% للاختبار مع تثبيت الحالة عند 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحجيم البيانات (Scaling) عشان الأرقام الكبيرة متلخبطش الموديل
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # بيتعلم ويحول بيانات التدريب
X_test = scaler.transform(X_test)      # بيحول بيانات الاختبار بنفس المعيار

# بناء الموديل (الغابات العشوائية) - وحش التوقعات!
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # هنا الموديل بيبدأ "يذاكر" العلاقة بين المواصفات والسعر

# --- 4. عرض النتائج والرسوم البيانية ---

# التوقع لبيانات الاختبار عشان نقيس الدقة
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

# رسم بياني للمقارنة بين الحقيقة والتوقع
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(y_test, y_pred, color='#1f77b4', alpha=0.4, label='Data Points')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Match')
ax.set_title("How accurate is our AI?")
ax.set_xlabel("Real Price")
ax.set_ylabel("AI Prediction")
ax.legend()
st.pyplot(fig)

# عرض الدقة في المربع الأخضر الكبير اللي طلبته
st.success(f"## 🎯 Model Accuracy: {score:.2%}")

# --- 5. واجهة توقع المستخدم (User Interactive Section) ---

st.markdown("---") # خط فاصل قبل منطقة المدخلات
st.subheader("🔮 Check Your Car's Value")
st.write("Fill in the details below to see our AI's estimation:")

# تقسيم المدخلات لعمودين عشان الشكل يكون شيك
col1, col2 = st.columns(2)

with col1:
    # القوائم المنسدلة بتسحب الاختيارات من البيانات الأصلية أوتوماتيك
    u_brand = st.selectbox("Select Brand & Model", df_raw['brand_model'].unique())
    u_year = st.number_input("Year of Manufacture (e.g. 2015)", 1990, 2025, 2018)
    u_km = st.number_input("Total Kilometers Driven", 0, 1000000, 40000)

with col2:
    u_fuel = st.selectbox("Fuel Type", df_raw['fuel'].unique())
    u_seller = st.selectbox("Seller Type", df_raw['seller_type'].unique())
    u_trans = st.selectbox("Transmission Type", df_raw['transmission'].unique())
    u_owner = st.selectbox("Previous Owners", df_raw['owner'].unique())

# زر التنفيذ
if st.button("Calculate Estimated Price 💰"):
    # خطوة التحويل: تحويل اختيارات المستخدم (النصوص) لأرقام بيفهمها الموديل
    # بنستخدم المترجمين اللي حفظناهم في le_dict
    try:
        user_input = pd.DataFrame({
            'year': [u_year],
            'km_driven': [u_km],
            'fuel': [le_dict['fuel'].transform([u_fuel])[0]],
            'seller_type': [le_dict['seller_type'].transform([u_seller])[0]],
            'transmission': [le_dict['transmission'].transform([u_trans])[0]],
            'owner': [le_dict['owner'].transform([u_owner])[0]],
            'brand_model': [le_dict['brand_model'].transform([u_brand])[0]]
        })

        # ترتيب الأعمدة عشان الموديل ميتهش
        user_input = user_input[X.columns]

        # عمل Scaling للمدخلات الجديدة بنفس معايير التدريب
        user_input_scaled = scaler.transform(user_input)

        # التوقع النهائي
        prediction = model.predict(user_input_scaled)

        # إظهار النتيجة مع شوية دلع (بالونات)
        st.balloons()
        st.info(f"### 💰 Estimated Price: {prediction[0]:,.2f} EGP")
        st.write("Note: This price is based on historical market data.")
        
    except Exception as e:
        st.error(f"Something went wrong! Error: {e}")
