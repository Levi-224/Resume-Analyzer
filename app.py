#pip install Flask PyPDF2 scikit-learn
#pip install panda numpy 
#pip install openai
from flask import Flask, render_template, request
import PyPDF2
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
import os
import openai

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        if 'upload_analyze' in request.form:
            file = request.files['resume']
            text = extract_text_from_pdf(file)
            predictions= predict_job(text)
            return render_template('index.html', predictions=predictions)

        elif 'use_openai' in request.form:
            file = request.files['resume']
            resume_text = extract_text_from_pdf(file)
            # Save the resume text to a temporary text file
            temp_file_path = 'resume.txt'
            if os.path.exists(temp_file_path) and os.stat(temp_file_path).st_size > 0:
                with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                    temp_file.write('')
            with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(resume_text)
            # Use OpenAI with the temporary text file
            chatgpt_response = use_openai(temp_file_path)
            # Delete the temporary text file after use
            os.remove(temp_file_path)

            return render_template('index.html', chatgpt_response=chatgpt_response)

    return render_template('index.html', predictions=None, chatgpt_response=None)


def extract_text_from_pdf(file):
    with open(file.filename, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

vectorizer = CountVectorizer(stop_words='english')

def predict_job(resume_content):
    
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df.isnull().sum()
    labels = df['Category']
    df['cleaned_resume']=df['Resume'].apply(lambda x:clean_function(x))
    inputs = df['cleaned_resume']
    inputs_train, inputs_test, Ytrain, Ytest = train_test_split(inputs, labels, random_state=123)
    Xtrain = vectorizer.fit_transform(inputs_train)
    Xtest = vectorizer.transform(inputs_test)
    model = RFC(max_depth = 15)
    model.fit(Xtrain, Ytrain)
    new_df=pd.Series(resume_content)
    pred = model.predict(new_inputs(new_df))
    return pred

def use_openai(file):
    from openai import OpenAI
    api_key_file = open('api.txt', 'r')
    openai.api_key = api_key_file.read()
    client=OpenAI(api_key=openai.api_key,)
    with open(file, 'r',encoding='utf-8') as file:
        resume_text = file.read()
    prompt=resume_text+"\n"+"Analyze the given resume text and from this predict 3 to 4 job opportunities suitable for this candidate. Provide insights into the candidates educational qualifications and also list out the skills of the candidate. The output should be in bulleted points with specific headings."
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "system", "content": "You are a resume analyser and can predict the jobs favourable for a resume from just analysing it and provide the predictions as bulleted points with specific headings."},
    {"role": "user", "content":prompt}
    ]
    )
    return response.choices[0].message.content

def new_inputs(resumes):
    cleaned_resumes = resumes.apply(lambda x:clean_function(x))
    transformed_resumes = vectorizer.transform(cleaned_resumes)
    return transformed_resumes


def clean_function(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText    

if  __name__ == '__main__':
    app.run(debug=True)
