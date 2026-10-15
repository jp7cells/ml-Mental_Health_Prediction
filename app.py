import joblib
import streamlit as st
import pandas as pd

pipeline = joblib.load("./mental_health_model.joblib")
st.set_page_config(page_title="แบบคัดกรองสุขภาพใจ", layout="centered")

CONFIDENCE_THRESHOLD = 0.6

CLASS_TH = {
    "Normal": "ปกติ",
    "Anxiety": "ภาวะวิตกกังวล",
    "Depression": "ภาวะซึมเศร้า",
    "Burnout": "ภาวะหมดไฟ",
}


def predict(inputs):
    features = pd.DataFrame([inputs])
    label = pipeline.predict(features)[0]
    proba = pipeline.predict_proba(features)[0]
    return label, dict(zip(pipeline.classes_, proba))


st.markdown("## แบบคัดกรองสุขภาพใจเบื้องต้นสำหรับนักศึกษา")
st.info(
    "เครื่องมือนี้เป็นระบบต้นแบบเพื่อการศึกษา ไม่ใช่การวินิจฉัยทางการแพทย์ "
    "โมเดลเรียนรู้จากข้อมูลจำลอง ผลลัพธ์ใช้เป็นแนวทางสำรวจตัวเองเท่านั้น"
)

st.markdown("#### การใช้ชีวิตประจำวัน")
col1, col2 = st.columns(2)
with col1:
    sleep_hours = st.slider("นอนเฉลี่ยกี่ชั่วโมงต่อคืน", 3.0, 12.0, 7.0, 0.5)
    social_media_hours = st.slider("ใช้โซเชียลมีเดียกี่ชั่วโมงต่อวัน", 0.0, 12.0, 3.0, 0.5)
with col2:
    physical_activity_days = st.slider("ออกกำลังกายกี่วันต่อสัปดาห์", 0, 7, 3)
    academic_work_pressure = st.slider("ความกดดันจากการเรียน/งาน (1 = ไม่มี, 10 = มากที่สุด)", 1, 10, 5)

st.markdown("#### ความรู้สึกช่วงนี้ (1 = น้อยที่สุด, 10 = มากที่สุด)")
col3, col4 = st.columns(2)
with col3:
    stress_level = st.slider("ความเครียด", 1, 10, 5)
    anxiety_score = st.slider("ความวิตกกังวล", 1, 10, 5)
    depression_score = st.slider("ความรู้สึกเศร้าหรือหมดหวัง", 1, 10, 5)
    mood_score = st.slider("อารมณ์โดยรวม (1 = แย่มาก, 10 = ดีมาก)", 1, 10, 5)
with col4:
    concentration_level = st.slider("สมาธิในการเรียน/ทำงาน", 1, 10, 5)
    work_life_balance = st.slider("สมดุลชีวิตกับการเรียน/งาน", 1, 10, 5)
    social_support = st.slider("มีคนรอบตัวให้พึ่งพาได้", 1, 10, 5)

if st.button("ดูผลการคัดกรอง", type="primary"):
    label, proba = predict({
        "sleep_hours": sleep_hours,
        "social_media_hours": social_media_hours,
        "academic_work_pressure": academic_work_pressure,
        "physical_activity_days": physical_activity_days,
        "stress_level": stress_level,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "work_life_balance": work_life_balance,
        "mood_score": mood_score,
        "concentration_level": concentration_level,
        "social_support": social_support,
    })
    confidence = proba[label]

    st.markdown("---")
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"ระบบไม่มั่นใจในผลลัพธ์ครั้งนี้ (ความมั่นใจ {confidence:.0%} "
            f"ต่ำกว่าเกณฑ์ {CONFIDENCE_THRESHOLD:.0%}) "
            "คำตอบของคุณมีลักษณะคาบเกี่ยวหลายกลุ่ม ควรพิจารณาร่วมกับผู้เชี่ยวชาญ"
        )
    elif label == "Normal":
        st.success(f"ผลการคัดกรอง: **{CLASS_TH[label]}** (ความมั่นใจ {confidence:.0%})")
    else:
        st.error(f"ผลการคัดกรอง: มีแนวโน้ม **{CLASS_TH[label]}** (ความมั่นใจ {confidence:.0%})")

    st.markdown("**ความน่าจะเป็นของแต่ละกลุ่ม**")
    chart = pd.DataFrame(
        {"ความน่าจะเป็น": [proba[c] for c in pipeline.classes_]},
        index=[CLASS_TH[c] for c in pipeline.classes_],
    )
    st.bar_chart(chart)

    if label != "Normal" or confidence < CONFIDENCE_THRESHOLD:
        st.markdown(
            "หากรู้สึกว่ากำลังเผชิญความยากลำบาก การพูดคุยกับคนที่ไว้ใจหรือผู้เชี่ยวชาญช่วยได้  \n"
            "• ศูนย์ให้คำปรึกษาของมหาวิทยาลัย  \n"
            "• สายด่วนสุขภาพจิต กรมสุขภาพจิต โทร **1323** (ตลอด 24 ชั่วโมง)"
        )
