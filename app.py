import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

# --- إعدادات الواجهة ---
st.set_page_config(page_title="AI Car Price Predictor", page_icon="🚗")
st.title("🚗 Smart Car Price Predictor (USD)")
st.markdown("---")

# --- 1. جلب البيانات ---
url = "https://raw.githubusercontent.com/PhilopateerDev/Car-Price-Project./main/Car%20details.csv"
df_raw = pd.read_csv(url)

# تحويل السعر للدولار
df_raw['selling_price'] = df_raw['selling_price'] / 84

st.subheader("📊 Historical Data Preview (Prices in USD)")
st.write(df_raw.head())

# --- 2. معالجة البيانات ---
df_raw['brand_model'] = df_raw['name'].str.split(' ').str.slice(0, 2).str.join(' ')
df = df_raw.copy()
le_dict = {}
categorical_cols = ['brand_model', 'fuel', 'seller_type', 'transmission', 'owner']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# --- التعديل الجوهري هنا ---
# بنختار الأعمدة اللي بقت أرقام فقط عشان الموديل ميغلطش
features = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'brand_model']
X = df[features] 
y = df['selling_price']

# --- 3. التدريب ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. عرض النتائج ---
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(y_test, y_pred, color='green', alpha=0.4)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
st.pyplot(fig)

st.success(f"## 🎯 Model Accuracy: {score:.2%}")

# --- 5. واجهة التوقع ---
st.markdown("---")
st.subheader("🔮 Predict in USD")

c1, c2 = st.columns(2)
with c1:
    u_brand = st.selectbox("Brand & Model", df_raw['brand_model'].unique())
    u_year = st.number_input("Year", 1990, 2025, 2018)
    u_km = st.number_input("Kilometers", 0, 1000000, 30000)
with c2:
    u_fuel = st.selectbox("Fuel", df_raw['fuel'].unique())
    u_seller = st.selectbox("Seller", df_raw['seller_type'].unique())
    u_trans = st.selectbox("Transmission", df_raw['transmission'].unique())
    u_owner = st.selectbox("Owner", df_raw['owner'].unique())

if st.button("Calculate Price In Dollars 💰"):
    # تحويل مدخلات المستخدم لأرقام
    input_df = pd.DataFrame({
        'year': [u_year],
        'km_driven': [u_km],
        'fuel': [le_dict['fuel'].transform([u_fuel])[0]],
        'seller_type': [le_dict['seller_type'].transform([u_seller])[0]],
        'transmission': [le_dict['transmission'].transform([u_trans])[0]],
        'owner': [le_dict['owner'].transform([u_owner])[0]],
        'brand_model': [le_dict['brand_model'].transform([u_brand])[0]]
    })
    
    # التأكد من نفس ترتيب الأعمدة
    input_df = input_df[features]
    input_sc = scaler.transform(input_df)
    res = model.predict(input_sc)
    
    st.balloons()
    st.info(f"### 💰 Estimated Price: ${res[0]:,.2f}")
