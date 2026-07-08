import streamlit as st
import pandas as pd
import joblib
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
}

.stApp{
    background:#0f1117;
    color:white;
}

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

.block-container{
    max-width:1100px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Hero */

.hero{
    text-align:center;
    margin-bottom:25px;
}

.hero h1{
    color:white;
    font-size:44px;
    font-weight:700;
    margin-bottom:8px;
}

.hero span{
    color:#4f8ef7;
}

.hero p{
    color:#9ba3b8;
    font-size:17px;
}

/* Divider */

hr{
    border:0;
    border-top:1px solid #2a2d3a;
    margin:25px 0;
}

/* Cards */

.card{

    background:#181c24;

    border:1px solid #2a2d3a;

    border-radius:14px;

    padding:25px;

}

/* Input */

div[data-testid="stNumberInput"] input{

    background:#111722 !important;

    color:white !important;

    border-radius:10px !important;

    border:1px solid #2a2d3a !important;

}

div[data-testid="stSelectbox"]>div>div{

    background:#111722 !important;

    color:white !important;

    border-radius:10px !important;

}

/* Button */

.stButton>button{

    width:100%;

    background:linear-gradient(90deg,#4f8ef7,#2c6be0);

    color:white;

    font-weight:600;

    border:none;

    border-radius:12px;

    padding:15px;

    font-size:17px;

}

.stButton>button:hover{

    background:linear-gradient(90deg,#5a97ff,#3d79eb);

}

/* Result Card */

.result{

    padding:25px;

    border-radius:15px;

    margin-top:25px;

}

.result.red{

    background:#2b1616;

    border:1px solid #b53b3b;

}

.result.green{

    background:#122418;

    border:1px solid #2ecc71;

}

/* Confidence */

.progress{

    width:100%;

    background:#222;

    height:10px;

    border-radius:50px;

    overflow:hidden;

    margin-top:10px;

}

.progress-fill{

    height:10px;

    background:#4f8ef7;

    border-radius:50px;

}

.reason{

    background:#111722;

    padding:15px;

    border-radius:10px;

    border:1px solid #2a2d3a;

    margin-top:18px;

}

.stat{

    display:inline-block;

    background:#181c24;

    padding:10px 18px;

    margin:8px;

    border-radius:10px;

    border:1px solid #2a2d3a;

}

.footer{

    text-align:center;

    margin-top:40px;

    color:#666;

    font-size:14px;

}

</style>
""",unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")

model = load_model()

# ============================================================
# TITLE
# ============================================================

st.markdown("""

<div class="hero">

<h1>📡 Churn <span>Intelligence</span></h1>

<p>Enter customer attributes below to predict churn likelihood in real time</p>

</div>

""",unsafe_allow_html=True)

st.markdown("<hr>",unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================

left,right=st.columns(2)

with left:

    age=st.number_input("Age",18,100,30)

    gender=st.selectbox("Gender",["Male","Female"])

    tenure=st.number_input("Tenure (Months)",0,100,12)

    usage=st.number_input("Usage Frequency",0,50,15)

    last=st.number_input("Last Interaction",0,100,10)

with right:

    subscription=st.selectbox(
        "Subscription Type",
        ["Basic","Standard","Premium"]
    )

    contract=st.selectbox(
        "Contract Length",
        ["Monthly","Quarterly","Annual"]
    )

    spend=st.number_input(
        "Total Spend",
        0.0,
        100000.0,
        5000.0,
        step=100.0
    )

    support=st.number_input(
        "Support Calls",
        0,
        20,
        2
    )

    payment=st.number_input(
        "Payment Delay",
        0,
        100,
        5
    )

st.markdown("<hr>",unsafe_allow_html=True)

predict=st.button("🚀 Predict Customer Churn")
# ============================================================
# PREDICTION
# ============================================================

if predict:

    input_df = pd.DataFrame({

        "Age":[age],
        "Gender":[gender],
        "Tenure":[tenure],
        "Usage Frequency":[usage],
        "Support Calls":[support],
        "Payment Delay":[payment],
        "Subscription Type":[subscription],
        "Contract Length":[contract],
        "Total Spend":[spend],
        "Last Interaction":[last]

    })

    start = time.time()

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    end = time.time()

    prediction_time = (end-start)*1000

    churn_prob = probability[1]*100
    stay_prob = probability[0]*100

    confidence = min(probability[prediction]*100,99.9)

    reasons=[]

    if support>=5:
        reasons.append("📞 High number of support calls")

    if payment>=10:
        reasons.append("💳 Frequent payment delays")

    if tenure<12:
        reasons.append("⏳ Customer is relatively new")

    if spend<1000:
        reasons.append("💰 Low overall spending")

    if subscription=="Basic":
        reasons.append("📦 Basic subscription plan")

    if contract=="Monthly":
        reasons.append("📅 Monthly contract")

    if usage<8:
        reasons.append("📉 Low service usage")

    if last>30:
        reasons.append("⌛ Long time since last interaction")

    if len(reasons)==0:
        reasons.append("Customer behaviour appears stable.")

    recommendation=[]

    if prediction==1:

        if support>=5:
            recommendation.append("📞 Contact customer immediately")

        if payment>=10:
            recommendation.append("💳 Offer flexible payment option")

        if contract=="Monthly":
            recommendation.append("📅 Encourage yearly contract")

        if subscription=="Basic":
            recommendation.append("⬆ Upgrade to Premium Plan")

        if tenure<12:
            recommendation.append("🎁 Offer welcome loyalty discount")

        recommendation.append("🎯 Send personalised retention offer")

        color="red"

        title="🔴 High Churn Risk"

        subtitle="This customer is likely to leave."

    else:

        recommendation.append("🌟 Customer is highly satisfied")

        recommendation.append("🎯 Continue loyalty rewards")

        recommendation.append("📧 Send appreciation email")

        if subscription!="Premium":
            recommendation.append("⬆ Recommend Premium plan")

        recommendation.append("💙 Maintain engagement")

        color="green"

        title="🟢 Low Churn Risk"

        subtitle="Customer is likely to stay."

    st.markdown(f"""
<div class="result {color}">

<h2>{title}</h2>

<p>{subtitle}</p>

<h3>Confidence Score</h3>

<h2>{confidence:.1f}%</h2>

<div class="progress">

<div class="progress-fill"
style="width:{confidence}%"></div>

</div>

</div>

""",unsafe_allow_html=True)

    st.markdown("### 📌 Why this Prediction?")

    for r in reasons:

        st.markdown(f"""
<div class="reason">

{r}

</div>

""",unsafe_allow_html=True)

    st.markdown("")

    st.markdown("### 💡 Recommended Actions")

    c1,c2=st.columns(2)

    for i,item in enumerate(recommendation):

        if i%2==0:

            with c1:
                st.success(item)

        else:

            with c2:
                st.success(item)

    st.markdown("")

    st.markdown("### 📊 Prediction Statistics")

    a,b,c,d=st.columns(4)

    with a:

        st.markdown(f"""
<div class="stat">

<b>Churn Probability</b><br>

{churn_prob:.2f}%

</div>

""",unsafe_allow_html=True)

    with b:

        st.markdown(f"""
<div class="stat">

<b>Retention Probability</b><br>

{stay_prob:.2f}%

</div>

""",unsafe_allow_html=True)

    with c:

        st.markdown(f"""
<div class="stat">

<b>Prediction Time</b><br>

{prediction_time:.2f} ms

</div>

""",unsafe_allow_html=True)

    with d:

        risk="High" if prediction==1 else "Low"

        st.markdown(f"""
<div class="stat">

<b>Risk Level</b><br>

{risk}

</div>

""",unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""

<div class="footer">

Customer Churn Prediction System

<br>

Built using Python • Streamlit • HistGradientBoostingClassifier

</div>

""",unsafe_allow_html=True)