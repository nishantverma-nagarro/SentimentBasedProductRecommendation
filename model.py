'''

#1. Connect to Google drive

#importing colab libraries
from google.colab import drive
drive.mount('/content/drive')

'''

#filepath = '/content/drive/MyDrive/Colab Notebooks/Capstone/'

#importing librariescls
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
import string
import pickle
import time

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# To Check the most word occurence using word cloud
from wordcloud import WordCloud ,STOPWORDS



from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,f1_score,precision_score,accuracy_score, plot_confusion_matrix
from sklearn.metrics import pairwise_distances
#from sklearn.model_selection import GridSearchCV
#from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

#import xgboost as xgb
#from xgboost import XGBClassifier


'''
3. Load Data
'''

#df = pd.read_csv(filepath + 'sample30.csv')
df = pd.read_csv('data/sample30.csv')
df.head(2)

'''
#df_attributes = pd.read_csv(filepath + 'Data+Attribute+Description.csv', encoding = "1250")
df_attributes = pd.read_csv('data/Data+Attribute+Description.csv', encoding = "1250")
df_attributes


4. Exploratory data analysis
'''

#data shape
df.shape

#data information
df.info()

#number of review
print("number of review texts:", len(df['reviews_text']))
print("number of review ratings:", len(df['reviews_rating']))
print("number of unique reviewer:", len(df['reviews_username'].unique()))

#look at unique values in each column
print('\nUnique values: \n', df.nunique())

'''
Outcome:

- We can see that are 5 values in ratings
- 2 different sentiments
- Will have to treat missing values
- Products belong to 200+ brands
- Products belong to 200+ manufacturer


Missing Values (%)
'''

#Null precentage
print("Percentage of missing values :")
print(df.isna().mean().round(4) * 100)

sns.set_theme(style="darkgrid")

'''
Analyse Did Purchase
'''

print('Did Purchase distribution (%)')
print(round(df['reviews_didPurchase'].value_counts()/len(df['reviews_didPurchase']) * 100,2))
print('percentage of Missing Values(%):', round(df['reviews_didPurchase'].isna().sum()/len(df['reviews_didPurchase']) * 100,2))

'''
"True" is only 4% of the total data which is very low. We will replace the missing values with "No Data" for our analysis
'''

plt.figure(figsize=(10,8))
ax = sns.countplot(df['reviews_didPurchase'],palette="Set3")
ax.set_xlabel(xlabel="Shoppers did purchase the product", fontsize=15)
ax.set_ylabel(ylabel='Count of did purchase', fontsize=15)
ax.tick_params(labelsize=13)
plt.show()


'''
Many people who have provided review have not purchased the product

Top products that were purchased:
'''
result = df[df['reviews_didPurchase'] == True]
print(result['name'].value_counts()[0:10])
result['name'].value_counts()[0:10].plot(kind = 'barh', figsize=[15,10], fontsize=15,color='Green').invert_yaxis()

'''
Top purchased brands
'''

print(result['brand'].value_counts()[0:10])
result['brand'].value_counts()[0:10].plot(kind = 'barh', figsize=[15,10], fontsize=15,color='Green').invert_yaxis()

'''
Analyse Did Recommend
'''

print('Did Recommend distribution (%)')
print(round(df['reviews_doRecommend'].value_counts()/len(df['reviews_doRecommend']) * 100,2))
print('percentage of Missing Values (%):', round(df['reviews_doRecommend'].isna().sum()/len(df['reviews_doRecommend']) * 100,2))

'''
Top products that were recommended:
'''

result = df[df['reviews_doRecommend'] == True]
print(result['name'].value_counts()[0:10])
result['name'].value_counts()[0:10].plot(kind = 'barh', figsize=[15,10], fontsize=15,color='Green').invert_yaxis()

'''
Analyse Ratings
'''

df["reviews_rating"].describe()

print('Rating distribution (%)')
print(round(df['reviews_rating'].value_counts()/len(df['reviews_rating']) * 100,2))

plt.figure(figsize=[10,5]) #[width, height]
x = list(df['reviews_rating'].value_counts().index)
y = list(df['reviews_rating'].value_counts())
plt.barh(x, y)
plt.title('Distribution of ratings', fontsize=20, weight='bold', color='navy', loc='center')
plt.xlabel('Count', fontsize=15, weight='bold', color='navy')
plt.ylabel('Ratings', fontsize=15, weight='bold', color='navy')
plt.show()

'''
Maximum rating given by users are 5 for any product.

Analyse User sentiment
'''

#user_sentiment
print('User Sentiment distribution (%)')
print(round(df['user_sentiment'].value_counts()/len(df['user_sentiment']) * 100,2))

'''
Top products with Positive sentiment
'''

result = df[df['user_sentiment'] == 'Positive']
print(result['name'].value_counts()[0:10])
result['name'].value_counts()[0:10].plot(kind = 'barh', figsize=[15,10], fontsize=15,color='Green').invert_yaxis()

'''
Top brand with Positive sentiment
'''

print(result['brand'].value_counts()[0:10])
result['brand'].value_counts()[0:10].plot(kind = 'barh', figsize=[15,10], fontsize=15,color='Green').invert_yaxis()

'''
Top products with Negative sentiment
'''

result = df[df['user_sentiment'] == 'Negative']
print(result['name'].value_counts()[0:10])
result['name'].value_counts()[0:10].plot(kind = 'barh', figsize=[15,10], fontsize=15,color='Magenta').invert_yaxis()


'''
Top band with negative sentiment
'''
print(result['brand'].value_counts()[0:10])
result['brand'].value_counts()[0:10].plot(kind = 'barh', figsize=[15,10], fontsize=15,color='Magenta').invert_yaxis()


'''
Analyse other columns
'''

df["manufacturer"].value_counts()[:10]

df["reviews_username"].value_counts()[:10]

plt.figure(figsize=(10,8))
ax = sns.histplot(hue=df['reviews_rating'],x=df['user_sentiment'])
ax.set_xlabel(xlabel="USer Sentiment", fontsize=17)
ax.set_ylabel(ylabel='Reviews Count', fontsize=17)
ax.axes.set_title('Review Segregation', fontsize=17)
ax.tick_params(labelsize=13)


stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=300, max_font_size=40,
                     scale=3, random_state=1).generate(str(df['reviews_text'].value_counts()))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


#word cloud for review where user sentiment was positive
wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=300, max_font_size=40,
                     scale=3, random_state=1).generate(str(df[df['user_sentiment'] == 'Positive']['reviews_text'].value_counts()))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#word cloud for review where user sentiment was negative
wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_words=300, max_font_size=40,
                     scale=3, random_state=1).generate(str(df[df['user_sentiment'] == 'Negative']['reviews_text'].value_counts()))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


'''
5. Data Preprocessing
Missing Values Treatment

'''

print("Missing Value Count:")
print(df.isnull().sum())


# Replace nulls in title will a blank space
df['reviews_title'].fillna('',inplace=True)

# Drop the columns with less than 20% of values
df.dropna(thresh = len(df) * .2, axis = 1, inplace = True)
print(df.isna().mean().round(4) * 100)

# In reviews_didPurchase, we will replace null values to False as infered in above analysis

df['reviews_didPurchase'] = df['reviews_didPurchase'].fillna('No Data')
df.reviews_didPurchase.unique()


#on the reviews_doRecommend column, replace null values with "No Data"
df['reviews_doRecommend'].fillna('No Data', inplace=True)

#Dropping the review date

df.drop('reviews_date', axis=1, inplace=True)

#<b>Fix the issue with user sentiment and rating</b>
# for correcting the user sentiment according to rating 
'''def review_sentiment_clear(x):
  if x >= 3 :
    return 'Positive'
  elif x > 0 and x < 3  :
    return 'Negative' '''


'''df['user_sentiment'] = df['reviews_rating'].apply(review_sentiment_clear)'''

'''plt.figure(figsize=(10,8))
ax = sns.histplot(hue=df['reviews_rating'],x=df['user_sentiment'])
ax.set_xlabel(xlabel="User Sentiment", fontsize=17)
ax.set_ylabel(ylabel='Reviews Count', fontsize=17)
ax.axes.set_title('Review Segregation', fontsize=17)
ax.tick_params(labelsize=13)'''


df.isna().sum()

df.dropna(inplace=True)
df.isna().sum()

df.shape

df['user_sentiment'].value_counts()

'''df['user_sentiment'].map({'Positive':1,'Negative':0})'''

#map the categorical user_sentiment to numerical 1 or 0 for modelling
df['user_sentiment'] = df['user_sentiment'].map({'Positive':1,'Negative':0})

df['user_sentiment'].value_counts()


'''
6. Text Preprocessing

'''

df["reviews_full"] = df[['reviews_title', 'reviews_text']].agg('. '.join, axis=1).str.lstrip('. ')

df[["reviews_full", "user_sentiment"]].sample(10)

#function to clean the text and remove all the unnecessary elements.
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub("\[\s*\w*\s*\]", "", text)
    dictionary = "abc".maketrans('', '', string.punctuation)
    text = text.translate(dictionary)
    text = re.sub("\S*\d\S*", "", text)
    
    return text


df_clean = df[['id','name', 'reviews_full', 'user_sentiment']]

df_clean["reviews_text"] = df_clean.reviews_full.apply(lambda x: clean_text(x))


# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
def remove_stopword(text):
    words = [word for word in text.split() if word.isalpha() and word not in stopwords]
    return " ".join(words)


lemmatizer = WordNetLemmatizer()
# Lemmatize the sentence
def lemma_text(text):
    word_pos_tags = nltk.pos_tag(word_tokenize(remove_stopword(text))) # Get position tags
    # Map the position tag and lemmatize the word/token
    words =[lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] 
    return " ".join(words)


df_clean["reviews_text_cleaned"] = df_clean.reviews_text.apply(lambda x: lemma_text(x))

df_clean.head()


'''
Check words after cleaning
'''

wordcloud = WordCloud(stopwords=stopwords,max_words=200).generate(str(df_clean.reviews_text_cleaned))

plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


def get_top_common_words(reviews, n_most_common):
    # flatten review column into a list of words, and set each to lowercase
    all_words = [word for review in reviews for word in 
                         review.lower().split()]


    # remove punctuation from reviews
    all_words = [''.join(char for char in review if \
                                 char not in string.punctuation) for \
                         review in all_words]


    # remove any empty strings that were created by this process
    all_words = [review for review in all_words if review]

    return Counter(all_words).most_common(n_most_common)


pos_reviews = df_clean[df_clean['user_sentiment']==1]
get_top_common_words(pos_reviews['reviews_text_cleaned'],10)


neg_reviews = df_clean[df_clean['user_sentiment']==0]
get_top_common_words(neg_reviews['reviews_text_cleaned'],10)


#function to collect the n-gram frequency of words

def get_top_n_ngram( corpus, n_gram_range ,n=None):
    vec = CountVectorizer(ngram_range=(n_gram_range, n_gram_range), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    #print(bag_of_words)
    sum_words = bag_of_words.sum(axis=0) 
    print("--1",sum_words)
    for word, idx in vec.vocabulary_.items():
        #print(word)
        #print(idx)
        break
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    #print("-31",words_freq)
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# display the top 10 words in the bigram frequency

common_words = get_top_n_ngram(pos_reviews['reviews_text_cleaned'], 2, 10)
pd.DataFrame(common_words)


# display the top 10 words in the bigram frequency

common_words = get_top_n_ngram(neg_reviews['reviews_text_cleaned'], 2, 10)
pd.DataFrame(common_words)


# display the top 10 words in the trigram frequency from entire data set

common_words = get_top_n_ngram(df_clean["reviews_text_cleaned"], 3, 10)
pd.DataFrame(common_words)


'''
7. Feature Extraction

'''

X = df_clean['reviews_text_cleaned']
y = df_clean['user_sentiment']


no_of_classes= len(pd.Series(y).value_counts())

for i in range(0,no_of_classes):
    print("Percent of {0}s: ".format(i), round(100*pd.Series(y).value_counts()[i]/pd.Series(y).value_counts().sum(),2), "%")
    

# using TF-IDF vectorizer using the parameters to get 650 features.


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=650, max_df=0.9, min_df=7, binary=True, 
                                   ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(df_clean['reviews_text_cleaned'])

y= df_clean['user_sentiment']


print(tfidf_vectorizer.get_feature_names())

#ickle.dump(tfidf_vectorizer,open(filepath+'pickle_file/tfidf_vectorizer.pkl','wb'))
pickle.dump(tfidf_vectorizer,open('pickle_file/tfidf_vectorizer.pkl','wb'))

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y, random_state=42, test_size=0.25)


'''
Class imbalance (using SMOTE)

The class difference between the positive and negative user sentiment is very high. Hence we need ti balance the 2 classes
'''

counter = Counter(y_train)
print('Before',counter)

sm = SMOTE()

# transform the dataset
X_train, y_train = sm.fit_resample(X_train, y_train)

counter = Counter(y_train)
print('After',counter)

'''
8. Sentiment Analysis Models

'''

# Function for Metrics
performance=[]

def model_metrics(y,y_pred,model_name,metrics):
  Accuracy = accuracy_score(y,y_pred)
  roc = roc_auc_score(y,y_pred)
  confusion = confusion_matrix(y,y_pred)
  precision = precision_score(y,y_pred)
  f1 = f1_score(y,y_pred)
  TP = confusion[1,1]  # true positive
  TN = confusion[0,0]  # true negatives
  FP = confusion[0,1]  # false positives
  FN = confusion[1,0]  # false negatives
  sensitivity= TP / float(TP+FN)
  specificity = TN / float(TN+FP)

  print("*"*50)
  print('Confusion Matrix =')
  print(confusion)
  print("sensitivity of the %s = %f" % (model_name,round(sensitivity,2)))
  print("specificity of the %s = %f" % (model_name,round(specificity,2)))
  print("Accuracy Score of %s = %f" % (model_name,Accuracy))
  print('ROC AUC score of %s = %f' % (model_name,roc))
  print("Report=",)
  print(classification_report(y,y_pred))
  print("*"*50)
  metrics.append(dict({'Model_name':model_name,
                       'Accuracy':Accuracy,
                       'Roc_auc_score':roc,
                       'Precision':precision,
                       'F1_score':f1}))
  return metrics




'''
Logistic Regresssion

'''

lr = LogisticRegression()
lr.fit(X_train, y_train)


y_pred_lr = lr.predict(X_train)
peformance = model_metrics(y_train,y_pred_lr,'Logistic Regression',performance)


# 1. Logsitic Regression 
for c in [0.01, 0.05, 0.25, 0.5, 1, 1.5, 2]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    cm = confusion_matrix(y_test, lr.predict(X_test))
    print('Sensitivity for C = {0} is {1}'.format(c, cm[1][1]/sum(cm[1])))
    print('Specificity for C = {0} is {1}'.format(c, cm[0][0]/sum(cm[0])))
    
    
final_lr = LogisticRegression(C=2)
final_lr.fit(X_train, y_train)


y_pred_lr_final = final_lr.predict(X_train)
peformance = model_metrics(y_train,y_pred_lr_final,'Logistic Regression - Tuned',performance)


feature_to_coef = {
    word: coef for word, coef in zip(
     tfidf_vectorizer.get_feature_names(), final_lr.coef_[0])
}

print('Positive Words')
for best_positive in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1],
    reverse=True)[:10]:
    print(best_positive)
    
print('Negative Words')
for best_negative in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1])[:10]:
    print(best_negative)

    
'''
Evaluate all the models with test data set

'''

#Evaluatopn between lr , rf, nb and boost 
test_performance=[]

y_test_pred_lr_final = final_lr.predict(X_test)
test_peformance = model_metrics(y_test,y_test_pred_lr_final,'Logistic Regression - Tuned',test_performance)

'''
Based on the train and test ROC AUC values the "Logistic Regression" model is the best.

Saving the Logistic Regression model

'''

#pickle.dump(lr,open(filepath+'pickle_file/model.pkl','wb'))
pickle.dump(lr,open('pickle_file/lr_model.pkl','wb'))


'''
Recommedation system

User and User recommedation system

1. Load Data

'''

#f = pd.read_csv(filepath + 'sample30.csv')
df = pd.read_csv('data/sample30.csv')
df.head(2)

'''
2. Data preprocessing

'''

df['reviews_rating'].describe()

ratings= df[['name', 'reviews_rating', 'reviews_username']]

ratings.head()

# Checking for null values
ratings.info()


ratings = ratings[~ratings.reviews_username.isna()]

ratings.info()

train,test = train_test_split(ratings,train_size=0.70,random_state=45)
print('train shape = ',train.shape)
print('test shape = ',test.shape)

train.name.nunique()

train.reviews_username.nunique()

# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(0)

df_pivot.head(3)


df_pivot.shape


'''
3. Creating dummy train & dummy test dataset

'''

# Copy the train dataset into dummy_train
dummy_train = train.copy()

# The products not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)


dummy_train.head()

dummy_train.shape

'''
4. User Similarity Matrix

Using Cosine Similarity

'''

user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

user_correlation.shape


'''
Using adjusted Cosine

'''

df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)

df_pivot.head()

'''
Normalising the rating of the movie for each user around 0 mean

'''

mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

df_subtracted.head()

'''
Finding cosine similarity

'''

user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


'''
Prediction:

'''

user_correlation.shape


# Removing negative rating
user_correlation[user_correlation<0]=0
user_correlation

user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings

user_predicted_ratings.shape

user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()

'''
Find the top 5 recommendation for the user

'''

user_input = str('00sab00') # for checking

# Recommended products for the selected user based on ratings
d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d


'''
Evaluation - User User

'''

# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape


common.head()

# convert into the user-product matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username', columns='name', 
                                              values='reviews_rating')

# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)

user_correlation_df['userId'] = df_subtracted.index
user_correlation_df.set_index('userId',inplace=True)
user_correlation_df.head()

common.head(1)

list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()


user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


user_correlation_df_1.shape

user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

user_correlation_df_3 = user_correlation_df_2.T
user_correlation_df_3.head()

user_correlation_df_3.shape

user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings

dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)

dummy_test.shape

common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)
common_user_predicted_ratings


'''
Find rmse :

'''

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)

common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


common_.head()

common_.info()

y

y.shape

'''
below analysis is because. i was getting error while calculating rmse

'''

y.sum()

common_.sum()

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

total_non_nan

((common_ - y )**2).info()

y

common_

'''y = np.nan_to_num(y)
print(y)'''

y.sum()

y.dtype


'''common_ = common_.fillna(0)'''

common_.info()

common_.sum()

common_ - y

common_.columns

common_.index

print(y.shape)

common_.shape

'''sum((common_ - y)**2)'''

(common_ - y).sum()

(common_ - y)**2

((common_ - y)**2).sum()

'''rmse_user = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse_user)'''

rmse_user = (sum(((common_ - y)**2).sum())/total_non_nan)**0.5
print(rmse_user)

'''
---------------------------------- End of User - User based similarity ----------------------------------

Selecting Recommendation system and Saving

User-User rmse: 2.52
Item-Item remse: 3.57

User-user based model is the better model. Saving the pickle file

'''

#pickle.dump(user_final_rating, open(filepath+'pickle_file/user_based_recomm_model.pkl','wb'))
pickle.dump(user_final_rating, open('pickle_file/user_based_recomm_model.pkl','wb'))


'''
Recommendation of Top 20 Products to a Specified User

# load all pkl files
tfidf_model = pickle.load(open(filepath+'pickle_file/tfidf_vectorizer.pkl', 'rb'))
user_based_recomm_model = pickle.load(open(filepath+'pickle_file/user_based_recomm_model.pkl', 'rb'))
LR_model = pickle.load(open(filepath+'pickle_file/model.pkl', 'rb'))

'''

# load all pkl files
tfidf_model = pickle.load(open('pickle_file/tfidf_vectorizer.pkl', 'rb'))
user_based_recomm_model = pickle.load(open('pickle_file/user_based_recomm_model.pkl', 'rb'))
LR_model = pickle.load(open('pickle_file/lr_model.pkl', 'rb'))

user = str('00sab00') # test user

# Recommend top 20 products
user_top20 = user_based_recomm_model.loc[user].sort_values(ascending=False)[:20]

user_top20 = pd.DataFrame(user_top20)  #.to_records())
user_top20.reset_index(inplace = True)
user_top20

'''
Cleaning the dataframe

'''
#df = pd.read_csv(filepath + 'sample30.csv')
df = pd.read_csv('data/sample30.csv')
df.head(2)

df['user_sentiment']= df['user_sentiment'].apply(lambda x:1 if x=='Positive' else 0)
df.head()

# Replace nulls 
df['reviews_title'].fillna('',inplace=True)

# merge reviews columns
df['reviews']=df['reviews_text']+df['reviews_title']
df.drop(['reviews_text','reviews_title'],axis=1,inplace=True)
df.head()

# df_clean -> cleaned columns for recommendation and sentiment models
df_clean = df[['name','reviews_username','reviews','reviews_rating','user_sentiment']]
df_clean.head()

df_clean.dropna(inplace=True)

df_clean.shape

#save the df_clean dataframe

#df_clean.to_csv(filepath + 'df_clean.csv')
df_clean.to_csv('data/df_clean.csv')

'''
Merge the clean data set with the recommended products for the user

'''

#df_clean = pd.read_csv(filepath + 'df_clean.csv')
df_clean = pd.read_csv('data/df_clean.csv')
df_clean.head(2)

df_clean.sample()

# merge top 20 products and its reviews
top20_products_setiment = pd.merge(user_top20,df_clean,on = ['name'])
top20_products_setiment.head()

'''
Pass 'top20_products' into tfidf model first and into sentiment model to find sentiment score.

'''

# convert text to feature
top20_products_tfidf = tfidf_model.transform(top20_products_setiment['reviews'])

# model prediction
top20_products_pred =LR_model.predict(top20_products_tfidf)
top20_products_pred

top20_products_setiment['top20_products_pred']=top20_products_pred

'''
senti_score is given by the percentage of positive reviews to the total reviews for each products.

'''

senti_score = top20_products_setiment.groupby(['name'])['top20_products_pred'].agg(['sum','count']).reset_index()
senti_score['percent'] = round((100*senti_score['sum'] / senti_score['count']),2)
senti_score.head()

senti_score

'''
Top 5 products:

'''

senti_score = senti_score.sort_values(by='percent',ascending=False)
senti_score

senti_score['name'].head().tolist()