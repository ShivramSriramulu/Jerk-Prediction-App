import streamlit as st
import pandas as pd
import joblib
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import base64
import streamlit as st

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def set_page_styles():
    background_image_base64 = get_image_base64('subreddit.png')
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: lavender;
        }}
        .header-background {{
            background-image: url("{background_image_base64}");
            background-size: cover;
            background-position: center;
            height: 250px;  # Adjust height to show the full image as needed
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    css_code = """
    <style>
    body {
  font-family: 'lato', sans-serif;
}
.container {
  max-width: 1000px;
  margin-left: auto;
  margin-right: auto;
  padding-left: 10px;
  padding-right: 10px;
}

h2 {
  font-size: 26px;
  margin: 20px 0;
  text-align: center;
  small {
    font-size: 0.5em;
  }
}

.responsive-table {
  li {
    border-radius: 3px;
    padding: 25px 30px;
    display: flex;
    justify-content: space-between;
    margin-bottom: 25px;
  }
  .table-header {
    background-color: #95A5A6;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .table-row {
    background-color: #ffffff;
    box-shadow: 0px 0px 9px 0px rgba(0,0,0,0.1);
  }
  .col-1 {
    flex-basis: 10%;
  }
  .col-2 {
    flex-basis: 40%;
  }
  .col-3 {
    flex-basis: 25%;
  }
  .col-4 {
    flex-basis: 25%;
  }
  
  @media all and (max-width: 767px) {
    .table-header {
      display: none;
    }
    .table-row{
      
    }
    li {
      display: block;
    }
    .col {
      
      flex-basis: 100%;
      
    }
    .col {
      display: flex;
      padding: 10px 0;
      &:before {
        color: #6C7A89;
        padding-right: 10px;
        content: attr(data-label);
        flex-basis: 50%;
        text-align: right;
      }
    }
  }
}
    .green-row {
    color: #57E796;
    }

.white-row {
    color: #777777 ;
}

.red-row {
    color: #FF5723;
}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)
    



nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def nltk_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def load_resources():
    @st.cache_resource
    def load_column_transformer():
        return joblib.load("column_transformer.joblib")

    @st.cache_resource
    def load_svd_transformer():
        return joblib.load("svd_transformer.joblib")

    @st.cache_resource
    def load_label_encoder():
        return joblib.load("label_encoder.joblib")

    @st.cache_resource
    def load_models():
        models = {name: joblib.load(f"{name}_model.joblib") for name in ['rf', 'lr', 'dt', 'svc', 'xgb']}
        return models

    @st.cache_resource
    def load_accuracies():
        with open('model_accuracies.json', 'r') as f:
            accuracies = json.load(f)
        return accuracies

    column_transformer = load_column_transformer()
    svd_transformer = load_svd_transformer()
    label_encoder = load_label_encoder()
    models = load_models()
    accuracies = load_accuracies()

    return column_transformer, svd_transformer, label_encoder, models, accuracies

def classify_text(input_text, resources):
    column_transformer, svd_transformer, label_encoder, models = resources[:4]
    df = pd.DataFrame([input_text], columns=['selftext'])
    X_transformed = column_transformer.transform(df)
    X_reduced = svd_transformer.transform(X_transformed)
    predictions = {model_name: label_encoder.inverse_transform(model.predict(X_reduced))[0] for model_name, model in models.items()}
    return predictions

def format_prediction(prediction):
    if prediction == 'ytj':
        return f'Yes 👍'
    elif prediction == 'ntj':
        return f'No 👼'
    elif prediction == 'na':
        return f'Not enough info 🤦'
    else:
        return f'{prediction}'
    
def get_row_color(prediction):
    if prediction == 'Yes 👍':
        return 'red-row'
    elif prediction == 'No 👼':
        return 'green-row'
    elif prediction == 'Not enough info 🤦':
        return 'white-row'
    else:
        return 'white-row'



def main():
    
    set_page_styles()
    st.markdown('<div class="header-background"></div>', unsafe_allow_html=True)
    st.markdown('', unsafe_allow_html=True)
    st.title('Am I The Jerk Prediction App')
    input_text = st.text_area("Enter your text here:", height=200)
    resources = load_resources()
    column_transformer, svd_transformer, label_encoder, models, accuracies = resources
   
    model_names = {
        'rf': 'Random Forest',
        'lr': 'Logistic Regression',
        'dt': 'Decision Tree',
        'svc': 'Support Vector Classifier',
        'xgb': 'Extreme Gradient Boost'
    }

    if st.button('Predict'):
        predictions = classify_text(input_text, resources)
        data = []
        for model_key, prediction in predictions.items():
            model_full_name = model_names[model_key]
            descriptive_prediction = format_prediction(prediction)
            accuracy = accuracies.get(model_key, "N/A")
            data.append([model_full_name, descriptive_prediction, f"{accuracy:.2f}"])

        df_results = pd.DataFrame(data, columns=['Model', 'Prediction', 'Accuracy'])
            
  
        body="""
        <div class="container">
        <ul class="responsive-table">
            <li class="table-header">
            <div class="col col-1">Model</div>
            <div class="col col-1">Prediction</div>
            <div class="col col-1">Accuracy</div>
            </li>
        """

        # Convert DataFrame to HTML table
        df_results_html = df_results.to_html(index=False, header=False, classes='table-row')
        df_results_html = df_results_html.replace('<tr>', '<li class="table-row">').replace('</tr>', '</li>')
        df_results_html = df_results_html.replace('<td>', '<div class="col col-1">').replace('</td>', '</div>')
        for index, row in df_results.iterrows():
            color_class = get_row_color(row['Prediction'])
            # Replace <li class="table-row"> with <li class="table-row green-row">
            df_results_html = df_results_html.replace(f'<div class="col col-1">{row["Prediction"]}</div>', f'<div class="col col-1 {color_class}">{row["Prediction"]}</div>')


        # Add HTML table rows to the existing HTML structure
        body += df_results_html

        # Close the HTML structure
        body += """
        </ul>
        </div>
        """
        st.markdown(body, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
