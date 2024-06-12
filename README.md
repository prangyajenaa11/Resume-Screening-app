# Resume-Screening-app
This project is a web-based application designed to classify resumes into predefined categories. It uses natural language processing (NLP) and machine learning techniques to analyze the content of resumes and predict their relevant job category.

## Features

- Upload resume files in `txt` or `pdf` format.
- Automatically preprocess and clean the resume text.
- Predict the job category using a pre-trained machine learning model.
- Display the predicted job category.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/resume-screening-app.git
    cd resume-screening-app
    ```

2. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Download necessary NLTK data:

    ```sh
    python -m nltk.downloader punkt stopwords
    ```

4. Ensure you have the pre-trained models (`clf.pkl` and `tfidf.pkl`) in the project directory.

### Running the Application

To start the Streamlit application, run:

```sh
streamlit run app.py
```

Open your web browser and navigate to the provided URL (typically `http://localhost:8501`).

## Usage

1. Upload your resume file using the file uploader in the web app.
2. The app will preprocess and clean the resume text.
3. The cleaned text will be transformed using the TF-IDF vectorizer.
4. The machine learning model will predict the job category.
5. The predicted category will be displayed on the web page.

## Project Structure

- `app.py`: Main application script.
- `tfidf.pkl`: Pre-trained TF-IDF vectorizer.
- `clf.pkl`: Pre-trained classification model.
- `UpdatedResumeDataSet.csv`: Dataset used for training the model.
- `Resume Screening with Python.ipynb`: Jupyter notebook for model training and analysis.
- `README.md`: Project documentation.

## Model Training

The model was trained using a dataset of resumes (`UpdatedResumeDataSet.csv`). The training process involves:

1. Preprocessing the resume text to remove unwanted characters and noise.
2. Transforming the text data into numerical features using TF-IDF vectorization.
3. Training a machine learning classifier to predict the job category based on the features.

For detailed steps, refer to the `Resume Screening with Python.ipynb` notebook.


## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework.
- [NLTK](https://www.nltk.org/) for natural language processing tools.
- [Scikit-learn](https://scikit-learn.org/) for machine learning utilities.

