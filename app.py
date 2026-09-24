import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# 1. LOAD TRAINED MODEL
# ==========================================================

with open("model/fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================================
# 2. LOAD TF-IDF VECTORIZER
# ==========================================================

with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==========================================================
# 3. PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# ==========================================================
# 4. TITLE
# ==========================================================

st.title("📰 Fake News Detector")

st.write(
    "Machine Learning Based News Classification System"
)

st.divider()


# ==========================================================
# 5. TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📰 News Detector",
        "📊 Model Performance",
        "📁 Dataset Information"
    ]
)


# ==========================================================
# TAB 1 - NEWS DETECTOR
# ==========================================================

with tab1:

    st.header("Check News")

    st.info(
        "Enter a news headline or article. "
        "The Machine Learning model will classify "
        "it as REAL, FAKE, or UNCERTAIN."
    )

    news_text = st.text_area(
        "Enter News Text",
        height=220,
        placeholder="Enter news headline or article here..."
    )


    # ======================================================
    # CHECK NEWS BUTTON
    # ======================================================

    if st.button(
        "🔍 Check News",
        use_container_width=True
    ):

        if news_text.strip() == "":

            st.warning(
                "⚠️ Please enter some news text."
            )

        else:

            # ----------------------------------------------
            # Convert text into TF-IDF
            # ----------------------------------------------

            news_tfidf = vectorizer.transform(
                [news_text]
            )


            # ----------------------------------------------
            # Prediction
            # ----------------------------------------------

            prediction = model.predict(
                news_tfidf
            )[0]


            # ----------------------------------------------
            # Probability
            # ----------------------------------------------

            probabilities = model.predict_proba(
                news_tfidf
            )[0]


            # ----------------------------------------------
            # Confidence
            # ----------------------------------------------

            confidence = max(
                probabilities
            )


            # ----------------------------------------------
            # Confidence threshold
            # ----------------------------------------------

            threshold = 0.70


            # ----------------------------------------------
            # Prediction Result
            # ----------------------------------------------

            st.subheader(
                "Prediction Result"
            )


            if confidence < threshold:

                st.warning(
                    "⚠️ UNCERTAIN RESULT"
                )

                st.write(
                    "The model is not confident enough "
                    "to classify this news as REAL or FAKE."
                )

            elif str(prediction).upper() == "REAL":

                st.success(
                    "✅ REAL NEWS"
                )

            else:

                st.error(
                    "❌ FAKE NEWS"
                )


            # ----------------------------------------------
            # Confidence
            # ----------------------------------------------

            st.write(
                f"### Model Confidence: "
                f"{confidence * 100:.2f}%"
            )

            st.progress(
                float(confidence)
            )


            # ----------------------------------------------
            # Show entered news
            # ----------------------------------------------

            with st.expander(
                "View Entered News"
            ):

                st.write(
                    news_text
                )


# ==========================================================
# TAB 2 - MODEL PERFORMANCE
# ==========================================================

with tab2:

    st.header(
        "📊 Model Performance"
    )

    st.write(
        "Performance of the trained Logistic Regression "
        "model on the test dataset."
    )


    # ------------------------------------------------------
    # LOAD DATASETS
    # ------------------------------------------------------

    fake_data = pd.read_csv(
        "data/Fake.csv"
    )

    real_data = pd.read_csv(
        "data/True.csv"
    )


    # Add labels

    fake_data["label"] = "FAKE"
    real_data["label"] = "REAL"


    # Combine datasets

    data = pd.concat(
        [fake_data, real_data],
        ignore_index=True
    )


    # Create text

    data["text"] = (
        data["title"].fillna("")
        + " "
        + data["text"].fillna("")
    )


    # Remove empty text

    data = data[
        data["text"].str.strip() != ""
    ]


    # Input and output

    X = data["text"]
    y = data["label"]


    # Same split used during training

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # TF-IDF transformation

    X_test_tfidf = vectorizer.transform(
        X_test
    )


    # Predictions

    predictions = model.predict(
        X_test_tfidf
    )


    # Accuracy

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Test Samples",
            len(y_test)
        )


    with col3:

        st.metric(
            "Training Samples",
            len(y_train)
        )


    st.divider()


    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["FAKE", "REAL"]
    )


    st.markdown(
        f"""
        <table style="width:100%; text-align:center;
        border-collapse:collapse;">

        <tr>
            <th style="border:1px solid gray; padding:12px;">
                Actual / Predicted
            </th>

            <th style="border:1px solid gray; padding:12px;">
                FAKE
            </th>

            <th style="border:1px solid gray; padding:12px;">
                REAL
            </th>
        </tr>

        <tr>
            <th style="border:1px solid gray; padding:12px;">
                FAKE
            </th>

            <td style="border:1px solid gray; padding:12px;">
                {cm[0][0]}
            </td>

            <td style="border:1px solid gray; padding:12px;">
                {cm[0][1]}
            </td>
        </tr>

        <tr>
            <th style="border:1px solid gray; padding:12px;">
                REAL
            </th>

            <td style="border:1px solid gray; padding:12px;">
                {cm[1][0]}
            </td>

            <td style="border:1px solid gray; padding:12px;">
                {cm[1][1]}
            </td>
        </tr>

        </table>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # ------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------

    st.subheader(
        "Classification Report"
    )


    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )


    st.write("### FAKE")

    st.write(
        f"Precision: {report['FAKE']['precision']:.3f}"
    )

    st.write(
        f"Recall: {report['FAKE']['recall']:.3f}"
    )

    st.write(
        f"F1-Score: {report['FAKE']['f1-score']:.3f}"
    )


    st.write("### REAL")

    st.write(
        f"Precision: {report['REAL']['precision']:.3f}"
    )

    st.write(
        f"Recall: {report['REAL']['recall']:.3f}"
    )

    st.write(
        f"F1-Score: {report['REAL']['f1-score']:.3f}"
    )


# ==========================================================
# TAB 3 - DATASET INFORMATION
# ==========================================================

with tab3:

    st.header(
        "📁 Dataset Information"
    )


    # Load datasets

    fake_data = pd.read_csv(
        "data/Fake.csv"
    )

    real_data = pd.read_csv(
        "data/True.csv"
    )


    total_fake = len(
        fake_data
    )

    total_real = len(
        real_data
    )

    total_news = (
        total_fake + total_real
    )


    # ------------------------------------------------------
    # DATASET METRICS
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total News",
            total_news
        )


    with col2:

        st.metric(
            "Fake News",
            total_fake
        )


    with col3:

        st.metric(
            "Real News",
            total_real
        )


    st.divider()


    # ------------------------------------------------------
    # DATASET DISTRIBUTION
    # ------------------------------------------------------

    st.subheader(
        "Dataset Distribution"
    )


    fake_percentage = (
        total_fake / total_news
    ) * 100


    real_percentage = (
        total_real / total_news
    ) * 100


    st.write(
        f"**FAKE:** {total_fake} "
        f"({fake_percentage:.2f}%)"
    )


    st.progress(
        float(fake_percentage / 100)
    )


    st.write(
        f"**REAL:** {total_real} "
        f"({real_percentage:.2f}%)"
    )


    st.progress(
        float(real_percentage / 100)
    )


    st.divider()


    # ------------------------------------------------------
    # TECHNOLOGIES
    # ------------------------------------------------------

    st.subheader(
        "Technologies Used"
    )


    st.write(
        """
        • Python

        • Pandas

        • Scikit-learn

        • TF-IDF Vectorization

        • Logistic Regression

        • Streamlit

        • Machine Learning
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Disclaimer: This system provides a machine-learning "
    "classification based on its training data. It does not "
    "independently verify the factual accuracy of news."
)