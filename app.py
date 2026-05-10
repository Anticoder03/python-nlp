import streamlit as st
import nltk
import time
import spacy

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# =========================
# DOWNLOAD REQUIRED DATA
# =========================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# =========================
# LOAD SPACY MODEL
# =========================
nlp = spacy.load("en_core_web_sm")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NLP Pipeline",
    page_icon="🧠",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("🧠 NLP Pipeline Dashboard")
st.write("Built using NLTK + spaCy + Streamlit")

# =========================
# SIDEBAR
# =========================
operation = st.sidebar.selectbox(
    "Select NLP Operation",
    [
        "Sentence Tokenization",
        "Word Tokenization",
        "Stemming",
        "Lemmatization",
        "Stemming vs Lemmatization",
        "Stopword Removal",
        "POS Tagging"
    ]
)

# =========================
# TEXT INPUT
# =========================
text = st.text_area(
    "Enter Text",
    height=250,
    value="Natural Language Processing is amazing."
)

# =========================
# SENTENCE TOKENIZATION
# =========================
if operation == "Sentence Tokenization":

    if st.button("Run Operation"):

        result = sent_tokenize(text)

        st.subheader("Sentences")
        st.write(result)

# =========================
# WORD TOKENIZATION
# =========================
elif operation == "Word Tokenization":

    if st.button("Run Operation"):

        result = word_tokenize(text)

        st.subheader("Words")
        st.write(result)

# =========================
# STEMMING
# =========================
elif operation == "Stemming":

    if st.button("Run Operation"):

        stemmer = PorterStemmer()

        start_time = time.time()

        words = word_tokenize(text)

        stemmed_words = [
            stemmer.stem(word)
            for word in words
        ]

        end_time = time.time()

        st.subheader("Stemmed Output")
        st.write(stemmed_words)

        st.success(
            f"Execution Time: {end_time - start_time:.5f} sec"
        )

        st.subheader("Unique Stemmed Words")
        st.write(set(stemmed_words))

# =========================
# LEMMATIZATION
# =========================
elif operation == "Lemmatization":

    if st.button("Run Operation"):

        lemmatizer = WordNetLemmatizer()

        start_time = time.time()

        words = word_tokenize(text)

        lemmatized_words = [
            lemmatizer.lemmatize(word)
            for word in words
        ]

        end_time = time.time()

        st.subheader("Lemmatized Output")
        st.write(lemmatized_words)

        st.success(
            f"Execution Time: {end_time - start_time:.5f} sec"
        )

        st.subheader("Unique Lemmatized Words")
        st.write(set(lemmatized_words))
        
# =========================
# STEMMING VS LEMMATIZATION
# =========================
elif operation == "Stemming vs Lemmatization":

    if st.button("Run Operation"):

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        words = word_tokenize(text)

        comparison_data = []

        for word in words:

            stemmed_word = stemmer.stem(word)
            lemmatized_word = lemmatizer.lemmatize(word)

            comparison_data.append({
                "Original Word": word,
                "Stemmed": stemmed_word,
                "Lemmatized": lemmatized_word
            })

        st.subheader("Comparison Result")

        st.table(comparison_data)

        st.markdown("---")

        st.markdown("""
        ### Key Difference

        - **Stemming**
            - Faster
            - Removes suffixes aggressively
            - May produce invalid words

        - **Lemmatization**
            - Slower
            - Uses dictionary meaning
            - Produces meaningful words
        """)

# =========================
# STOPWORD REMOVAL
# =========================
elif operation == "Stopword Removal":

    if st.button("Run Operation"):

        stopword_list = set(stopwords.words('english'))

        words = word_tokenize(text)

        filtered_words = [
            word for word in words
            if word.lower() not in stopword_list
        ]

        st.subheader("Filtered Words")
        st.write(filtered_words)

# =========================
# POS TAGGING
# =========================
elif operation == "POS Tagging":

    if st.button("Run Operation"):

        doc = nlp(text)

        st.subheader("POS Tagging Result")

        for token in doc:

            st.markdown(f"""
            ### {token.text}

            - POS: `{token.pos_}`
            - TAG: `{token.tag_}`
            - Explanation: `{spacy.explain(token.tag_)}`
            """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("NLP Dashboard using Streamlit")