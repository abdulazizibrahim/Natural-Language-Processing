'''
    Name: Abdul Aziz Muhammad Ibrahim Isa
    Roll No: P17-6143
    Natural Language Processing Assignment # 4
    Sentiment Analysis of IMDB dataset with comparison with MultinomialNB Classifier
'''
import time
import pandas as pd
from nltk.corpus import wordnet as wnet
from nltk.corpus import sentiwordnet as snet
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
class SentimentAnalysis:
    def __init__(self, path):
        self.corpus = pd.read_csv(path)
        self.review = self.corpus.iloc[: , [0]] .values
        self.labels = self.corpus.iloc[:,1].values
        self.senti = []
        #print(self.labels)
    def ArrangeData(self):
        x = self.review.flatten()
        self.review = x.tolist()
        #print(self.review[1], "\n\n")
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
        #print(self.review[1])
    def SentiAnalysis(self):
        Flist = self.review
        for i in Flist:
            score = 0
            i = i.split(' ')
            for j in i:
                syn = wnet.synsets(j)
                if len(syn) > 0:
                    syn = syn[0]
                    senti = snet.senti_synset(syn.name())
                    score += senti.pos_score() - senti.neg_score()
            if score > 0:
                scorex = 'positive'
            else:
                scorex = 'negative'
            self.senti.append(scorex)
        acc = accuracy_score(self.senti, self.labels)
        #print(len(self.senti))
        print("accuracy of sentiment Analysis is : ", acc)
        #print(self.senti[0],"\n\n", self.senti[5],'\n\n', self.senti[4999])
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
    def menu(self):
        self.ArrangeData();
        self.DataPreprocessing()
        self.SentiAnalysis()
        self.TfidfVectorizers()
        self.MultinomialNBClassifier()


if __name__ == "__main__":
    obj = SentimentAnalysis('C:\\Users\\abdul\\Downloads\\imdb-dataset-of-50k-movie-reviews\\Dataset.csv')
    obj.menu()
