import nltk 
import pandas as pd
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import   PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn as sns

from wordcloud import WordCloud

#importing the CSV file
df = pd.read_csv('/kaggle/input/amazon-product-reviews/Reviews.csv')
df_copy =df.copy()  # to make edits we need to make a copy of the original dataframe

# Data Preprocessing
df_cleaned = df_copy.dropna().reset_index(drop=True) 
stop_words=set(stopwords.words("english"))  # set of stop words in english
text = df_cleaned['Text']
text = text.str.lower() # convert to lower case
tokenized_text = text.apply(word_tokenize)  # tokenization
df_cleaned['Text_token']=tokenized_text

df_cleaned['filterd'] = df_cleaned['Text_token'].apply(lambda x:', '.join(word for word in x if word not in stop_words) ) # removing stop workds

ps = PorterStemmer()
df_cleaned['Stemmed']= df_cleaned['filterd'].apply(lambda x:''.join(ps.stem(word) for word in x ) ) # stemming, not very effective in this case but done for practice

lem = WordNetLemmatizer()
df_cleaned['lematized']= df_cleaned['Stemmed'].apply(lambda x:''.join(lem.lemmatize(word,"v") for word in x ) ) # lemmatization, more effective than stemming

df_cleaned['lable'] = df_cleaned['Score'].apply(lambda x: 1 if x > 3 else -1 if x < 3 else 0)

# Model Training and Evaluation
x_train,x_test,y_train, y_test = train_test_split(df_cleaned['lematized'],df_cleaned['lable'],test_size = 0.3, random_state=42)

# TF-IDF Vectorization // making a matrix of words and their importance, numarical representation of text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

clf = LogisticRegression(max_iter=5000,class_weight='balanced') # balanced to overcome the big bias on the data
clf.fit(x_train_vec, y_train)

y_pred = clf.predict(x_test_vec)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(classification_report(y_test, y_pred))


# WordCloud for reviews

positive_text = " ".join(x_test[y_pred==1])
negative_text = " ".join(x_test[y_pred==-1])
norm_text = " ".join(x_test[y_pred ==0])

wordcloud_pos = WordCloud(background_color="white").generate(positive_text)
wordcloud_neg = WordCloud(background_color="black").generate(negative_text)
wordcloud_nor = WordCloud(background_color="blue").generate(norm_text)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.title("Positive Reviews")

plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plt.imshow(wordcloud_nor, interpolation='bilinear')
plt.axis("off")
plt.title("Normal Reviews")

plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.title("Negative Reviews")
plt.show()

