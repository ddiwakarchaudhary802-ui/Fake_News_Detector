import pickle


# Load trained model
with open("model/fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Take news from user
news_text = input("Enter news: ")


# Convert news into TF-IDF
news_tfidf = vectorizer.transform([news_text])


# Predict
prediction = model.predict(news_tfidf)[0]


# Show result
print()
print("-----------------------------")

if str(prediction).upper() == "REAL":
    print("Prediction: REAL NEWS")
else:
    print("Prediction: FAKE NEWS")

print("-----------------------------")