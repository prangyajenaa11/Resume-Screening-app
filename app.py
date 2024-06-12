import streamlit as st
import pickle
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained models
classifier = pickle.load(open('clf.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

def preprocess_text(text):
    """Clean resume text by removing URLs, mentions, hashtags, and special characters."""
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def main():
    st.title("Resume Classifier Application")
    uploaded_file = st.file_uploader('Upload your resume', type=['txt', 'pdf'])

    if uploaded_file:
        try:
            resume_content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            resume_content = uploaded_file.read().decode('latin-1')

        cleaned_resume = preprocess_text(resume_content)
        input_vector = tfidf_vectorizer.transform([cleaned_resume])
        prediction_id = classifier.predict(input_vector)[0]

        category_dict = {
            0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain", 4: "Business Analyst",
            5: "Civil Engineer", 6: "Data Science", 7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
            10: "ETL Developer", 11: "Electrical Engineering", 12: "HR", 13: "Hadoop", 14: "Health and fitness",
            15: "Java Developer", 16: "Mechanical Engineer", 17: "Network Security Engineer", 18: "Operations Manager",
            19: "PMO", 20: "Python Developer", 21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing"
        }

        category_name = category_dict.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
