# Flight Delay Prediction App
# 3 questions, Random Forest models. Trains once on load (cached).
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

st.set_page_config(page_title="Flight Delay Prediction", page_icon="✈️", layout="centered")

MONTHS = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']

# ---------------------------------------------------------------- load + train
@st.cache_resource(show_spinner="Getting the models ready (first time only)…")
def load_and_train():
    fd = pd.read_csv('flights_dataset.csv')
    fd['FL_DATE'] = pd.to_datetime(fd['FL_DATE'], format='mixed', errors='coerce')
    fd['fl_year'] = fd['FL_DATE'].dt.year
    fd['fl_month_num'] = fd['FL_DATE'].dt.month
    fd.columns = fd.columns.str.lower()

    def clean_airline(n):
        if pd.isna(n): return n
        n = str(n).strip().lower()
        for s in [' air lines',' airways',' airlines']: n = n.replace(s,'')
        return n + '-air'
    fd['airline'] = fd['airline'].apply(clean_airline)

    try:
        import airportsdata
        ap = airportsdata.load('IATA')
        fd['origin'] = fd['origin'].map(lambda x: ap[x]['city'].lower() if x in ap else x)
        fd['dest']   = fd['dest'].map(lambda x: ap[x]['city'].lower() if x in ap else x)
    except Exception:
        pass

    fd = fd[['fl_year','fl_month_num','airline','origin','dest','dep_delay','arr_delay']]
    fd = fd.dropna().drop_duplicates()
    fd['fl_month'] = fd['fl_month_num'].apply(lambda m: MONTHS[int(m)-1])

    # answer columns
    fd['is_delayed']    = (fd['arr_delay'] > 15).astype(int)
    fd['severity']      = fd['arr_delay'].apply(
        lambda x: 'on_time' if x <= 15 else ('minor' if x <= 60 else 'major'))
    fd['big_dep_delay'] = (fd['dep_delay'] > 15).astype(int)

    def pre(cat, num):
        return ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
            ('numerical', 'passthrough', num)])
    def rf():
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    scores = {}

    # Q1 - delayed yes/no
    X = fd[['airline','origin','dest','dep_delay','fl_year','fl_month']]
    y = fd['is_delayed']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    m1 = Pipeline([('p',pre(['airline','origin','dest','fl_month'],['dep_delay','fl_year'])),('m',rf())]).fit(Xtr,ytr)
    p = m1.predict(Xte)
    scores['q1'] = (accuracy_score(yte,p)*100, precision_score(yte,p)*100)

    # Q2 - severity
    y2 = fd['severity']
    X2tr,X2te,y2tr,y2te = train_test_split(X,y2,test_size=0.2,random_state=42,stratify=y2)
    m2 = Pipeline([('p',pre(['airline','origin','dest','fl_month'],['dep_delay','fl_year'])),('m',rf())]).fit(X2tr,y2tr)
    p = m2.predict(X2te)
    scores['q2'] = (accuracy_score(y2te,p)*100,
                    precision_score(y2te,p,average='weighted',zero_division=0)*100)

    # Q3 - big departure delay
    X3 = fd[['airline','origin','dest','arr_delay','fl_year','fl_month']]
    y3 = fd['big_dep_delay']
    X3tr,X3te,y3tr,y3te = train_test_split(X3,y3,test_size=0.2,random_state=42,stratify=y3)
    m3 = Pipeline([('p',pre(['airline','origin','dest','fl_month'],['arr_delay','fl_year'])),('m',rf())]).fit(X3tr,y3tr)
    p = m3.predict(X3te)
    scores['q3'] = (accuracy_score(y3te,p)*100, precision_score(y3te,p)*100)

    opts = dict(
        airlines=sorted(fd['airline'].unique()),
        origins=sorted(fd['origin'].unique()),
        dests=sorted(fd['dest'].unique()),
        years=sorted(fd['fl_year'].unique()),
    )
    return m1, m2, m3, opts, scores

m1, m2, m3, opts, scores = load_and_train()

# ---------------------------------------------------------------- UI
st.title("✈️ Flight Delay Prediction")
st.write("Pick the flight details and get a prediction. Three questions, three models.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Arrive late?", "📊 How bad?", "🛫 Big departure delay?", "📈 Model scores"])

def common_inputs(key):
    airline = st.selectbox("Airline", opts['airlines'], key=f"a{key}")
    c1, c2 = st.columns(2)
    origin = c1.selectbox("From (origin)", opts['origins'], key=f"o{key}")
    dest   = c2.selectbox("To (destination)", opts['dests'], key=f"d{key}")
    c3, c4 = st.columns(2)
    year  = c3.selectbox("Year", opts['years'], key=f"y{key}")
    month = c4.selectbox("Month", MONTHS, key=f"m{key}")
    return airline, origin, dest, year, month

# ---- TAB 1: delayed yes/no ----
with tab1:
    st.subheader("Will the flight arrive more than 15 minutes late?")
    airline, origin, dest, year, month = common_inputs("1")
    dep = st.number_input("Departure delay so far (minutes, can be negative if early)",
                          -60, 600, 10, key="dep1")
    if st.button("Predict", type="primary", key="b1"):
        row = pd.DataFrame([{'airline':airline,'origin':origin,'dest':dest,
                             'dep_delay':dep,'fl_year':year,'fl_month':month}])
        pred = m1.predict(row)[0]
        prob = m1.predict_proba(row)[0][1] * 100
        if pred == 1:
            st.error(f"⚠️ Likely LATE  ({prob:.0f}% chance of delay)")
        else:
            st.success(f"✅ Likely ON TIME  ({prob:.0f}% chance of delay)")

# ---- TAB 2: severity ----
with tab2:
    st.subheader("How bad will the delay be?")
    airline, origin, dest, year, month = common_inputs("2")
    dep = st.number_input("Departure delay so far (minutes)", -60, 600, 10, key="dep2")
    st.caption("Tip: 0–15 = on time, 15–60 = minor, 60+ = major. Change the number and press Predict.")
    if st.button("Predict", type="primary", key="b2"):
        row = pd.DataFrame([{'airline':airline,'origin':origin,'dest':dest,
                             'dep_delay':dep,'fl_year':year,'fl_month':month}])
        pred = m2.predict(row)[0]
        label = {'on_time':'✅ On time (15 min or less)',
                 'minor':'🟡 Minor delay (15–60 min)',
                 'major':'🔴 Major delay (over 60 min)'}[pred]
        st.subheader(label)
        proba = dict(zip(m2.classes_, m2.predict_proba(row)[0]))
        st.write("Chances for each:")
        st.bar_chart(pd.Series({
            'On time': proba.get('on_time', 0),
            'Minor':   proba.get('minor', 0),
            'Major':   proba.get('major', 0),
        }))

# ---- TAB 3: big departure delay ----
with tab3:
    st.subheader("Will the flight leave more than 15 minutes late?")
    airline, origin, dest, year, month = common_inputs("3")
    arr = st.number_input("Expected arrival delay (minutes)", -60, 600, 10, key="arr3")
    if st.button("Predict", type="primary", key="b3"):
        row = pd.DataFrame([{'airline':airline,'origin':origin,'dest':dest,
                             'arr_delay':arr,'fl_year':year,'fl_month':month}])
        pred = m3.predict(row)[0]
        prob = m3.predict_proba(row)[0][1] * 100
        if pred == 1:
            st.error(f"⚠️ Likely a BIG departure delay  ({prob:.0f}% chance)")
        else:
            st.success(f"✅ Departure likely OK  ({prob:.0f}% chance of big delay)")

# ---- TAB 4: scores ----
with tab4:
    st.subheader("How good are the models? (tested on unseen data)")
    table = pd.DataFrame({
        'Question': ['Q1 – Arrive late? (yes/no)',
                     'Q2 – How bad? (3 types)',
                     'Q3 – Big departure delay? (yes/no)'],
        'Accuracy %':  [round(scores['q1'][0],1), round(scores['q2'][0],1), round(scores['q3'][0],1)],
        'Precision %': [round(scores['q1'][1],1), round(scores['q2'][1],1), round(scores['q3'][1],1)],
    })
    st.table(table)
    st.caption("These are Random Forest results on a held-out test set (data the model never saw while training).")
