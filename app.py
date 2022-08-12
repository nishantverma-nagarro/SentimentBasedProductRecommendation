import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load models

tfidf_model = pickle.load(open('pickle_file/tfidf_vectorizer.pkl', 'rb'))
user_based_recomm_model = pickle.load(open('pickle_file/user_based_recomm_model.pkl', 'rb'))
LR_model = pickle.load(open('pickle_file/lr_model.pkl', 'rb'))
df_clean = df_clean = pd.read_csv('data/df_clean.csv')


def getResults(user):
    if user not in user_based_recomm_model.index:
        return 'Enter valid user name: ' + user

    # Recommend top 20 products
    user_top20 = user_based_recomm_model.loc[user].sort_values(ascending=False)[:20]
    user_top20 = pd.DataFrame(user_top20)  #.to_records())
    user_top20.reset_index(inplace = True)
    # user_top20

    top20_products_setiment = pd.merge(user_top20,df_clean,on = ['name'])

    # convert text to feature
    top20_products_tfidf = tfidf_model.transform(top20_products_setiment['reviews'])

    # model prediction
    top20_products_pred =LR_model.predict(top20_products_tfidf)

    top20_products_setiment['top20_products_pred']=top20_products_pred

    senti_score = top20_products_setiment.groupby(['name'])['top20_products_pred'].agg(['sum','count']).reset_index()
    senti_score['percent'] = round((100*senti_score['sum'] / senti_score['count']),2)

    # Top 5 product
    senti_score = senti_score.sort_values(by='percent',ascending=False)
    return('Top 5 recommended products {}'.format(senti_score['name'].head().tolist()))

@app.route("/", methods=['POST','GET'])
def home():
    if (request.method == 'POST'):
        app.logger.info("Selected user: " + request.form['UserID'])
        user = request.form['UserID']
        return render_template('index.html',prediction_text=(getResults(user)))

    else:
        return render_template("index.html")


    
if __name__ == "__main__":
    app.run(debug=True)