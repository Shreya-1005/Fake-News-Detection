import streamlit as st
import joblib as jb

vectorizer = jb.load("vectorizer.jb")
model = jb.load("lr_model.jb")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #203f9a;
            background: linear-gradient(120deg, #203f9a, #94c2da);
        }
        .title {
            text-align: center;
            color: #85a7a8;
            font-size: 40px;
            font-weight: bold;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #efe8e0;
        }
        .stTextArea label {
            font-size: 18px;
            color: #efe8e0;
            font-weight: bold;
        }
        .stButton button {
            background: linear-gradient(135deg, #ff7eb3, #ff758c);
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #ff758c, #ff4b4b);
            color:white;
            transform: scale(1.05);
        }
        .sidebar .css-1d391kg {
            background-color: #94c2da;
        }
        /* Success & Error Message */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.9);
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 0px;
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Detect whether a news article is <b>Real</b> or <b>Fake</b> instantly!</p>", unsafe_allow_html=True)

# Instruction Sidebar
st.sidebar.title("🔍 How It Works")
st.sidebar.success("""
✅ Enter a news article in the text box.  
✅ Click **Check News** to analyze.  
✅ Get an instant result!  
""")

news_input = st.text_area("✍️ Enter News Article Here:", "", height=250)

if st.button("🔍 Check News"):
    with st.spinner("🔎 Analyzing the article... Please wait!"):
        if news_input.strip():
            transformed_input = vectorizer.transform([news_input])
            prediction = model.predict(transformed_input)


            # Display results
            if prediction[0] == 1:
                st.success("✅ **The News is Real! 📰**")
                st.balloons() 
            else:
                st.error("🚨 **Warning! The News is Fake! ❌**")
                st.warning("⚠️ Be cautious while sharing information online!")
        else:
            st.warning("⚠️ Please enter some text to analyze!")
