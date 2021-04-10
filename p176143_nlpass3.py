'''
Name : Abdul Aziz Muhammad Ibrahim Isa
Roll No: P17-6143
Natural Language Processing Assignment # 3
'''
# all imports come here
import time
import pandas as pd
from sklearn.feature_extraction import stop_words
from string import punctuation as punc
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
stopWordsList=list(stop_words.ENGLISH_STOP_WORDS)
wl = WordNetLemmatizer()
ps = PorterStemmer()
class DataAnalysis:
    def __init__(self, path):
        self.corpus = pd.read_csv(path)
        self.review = self.corpus.iloc[: , [0]] .values
        self.labels = self.corpus.iloc[:,1].values
        self.ArrangeData()
    def ArrangeData(self):
        x = self.review.flatten()
        self.review = x.tolist()
        #print(self.review)
    def DataPreprocessing(self):
        tempList, stemmingList, finalList, tempSplit = [],[],[],[]
        joinOn = " "
        for rev in self.review:
            tempStr=rev.replace('<br /><br />','')
            tempList=tempStr.split(" ")
            remStopWord=[word for word in tempList if word not in stopWordsList] 
            remPunctuation=[word for word in remStopWord if word not in punc]
            for word in remPunctuation:
                stemmingList.append(wl.lemmatize(ps.stem(word),'v')) 
            joinStr=joinOn.join(stemmingList)
            finalList.append(joinStr)      
            stemmingList=[]
        self.review = finalList
    def TfidfVectorizers(self):
        vector = TfidfVectorizer(lowercase = True )
        vec = vector.fit_transform(self.review)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split (vec, self.labels,shuffle = True,  train_size = 0.7)
    def MultinomialNBClassifier(self):
        start_time=time.time()
        clf = MultinomialNB()
        clf.fit(self.train_x, self.train_y)
        label = clf.predict(self.test_x)
        total_time=time.time() - start_time
        acc = accuracy_score(self.test_y, label)
        print("Multnomial Naive Bayes Classifier : \n","Time Taken : %s seconds" % total_time,"\n Accuracy Score : ", acc )
    def LinearClassifiers(self):
        start_time=time.time()
        clf=SGDClassifier()
        clf.fit(self.train_x, self.train_y)
        label = clf.predict(self.test_x)
        acc = accuracy_score(self.test_y, label)
        total_time=time.time() - start_time
        print("Linear Classifier : \n","Time Taken : %s seconds" % total_time,"\n Accuracy Score : ", acc )
    def main(self):
        self.DataPreprocessing()
        self.TfidfVectorizers()
        self.LinearClassifiers()
        self.MultinomialNBClassifier()
if __name__ == "__main__":  
    obj = DataAnalysis('C:\\Users\\abdul\\Downloads\\imdb-dataset-of-50k-movie-reviews\\IMDB Dataset.csv')
    obj.main()
