import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

#Set Random seed to save the accuracy
np.random.seed(1000)

# Add the Data using pandas
DataSet = pd.read_csv(r"reviews.csv",encoding='latin-1')

# Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms

# Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
DataSet['comment'] = [entry.lower() for entry in DataSet['comment']]

# Step - 1c : Tokenization : In this each entry in the DataSet will be broken into set of words
DataSet['comment']= [word_tokenize(entry) for entry in DataSet['comment']]

# Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default
# it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(DataSet['comment']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    DataSet.loc[index, 'comment_final'] = str(Final_words)


# Step - 2: Split the model into Train and Test Data set
Train_X, Test_X, Train_Y, Test_Y = model_selection\
    .train_test_split(DataSet['comment_final'], DataSet['label'], test_size=0.3)


# Step - 3: Vectorize the words by using TF-IDF Vectorizer term frequency–inverse document frequency
#  - This is done to find how important a word in document is in comaprison to the DataSet
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(DataSet['comment_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# Step - 4: Now we can run different algorithms to classify out data check for accuracy


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3,gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)