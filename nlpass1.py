#assignment 1 Natural Language Processing
#Name : Abdul Aziz Muhammad Ibrahim Isa
#Roll No: P17-6143
#section CS-6A
#Submitted to Dr. Taimoor khan
#all imports come here
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation as punc
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import pandas as panda
wl = WordNetLemmatizer()
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
data = open('C:\\Users\\abdul\\Downloads\\Movies_TV.txt').read()
corpus = data
data = data.split('\n')
data.remove(data[-1])

def text_mining():
    rv = []
    for i in data:
        _, _, _, review = i.split('\t')
        rv.append(review)
        
    #task 2
    #removing unwanted whitespaces
    lengthofreviews = len(rv)
    avgrv, avgtk = 0,0
    data_1, wordx = [], []
    count, total, tokens = 0,0,0
    vocabA, vocabB = 0,0

    for i in rv:
        if(count != 0):
            vocabB += len(set(i))
            avgrv += len(i)
            i = i.strip()
        
            #normalisation of text
            i = i.lower()
        
            #removing punctuations
            for p in punc:
                i = i.replace(p, '')
            
                #removing stopwords
            words = word_tokenize(i)             
            words = [word for word in words if word not in stop_words]
        
            #stemming words and lemmatizing words
            for word in words:
                avgtk += len(word)
                wordx.append(wl.lemmatize(SnowballStemmer("english").stem(word)))
                
            tokens += len(words)
            vocabA += len(set(words))
        if(count == 1):
            unigrams = list(ngrams(words, 1))
            bigrams = list(ngrams(words, 2))
            trigrams = list(ngrams(words, 3))
            total = len(words)
        
        count += 1
  
    print("unigrams", unigrams, "\n\n")
    print("bigrams", bigrams,"\n\n")
    print("trigrams", trigrams,"\n\n")

    print( "probability of unigram :" , len(unigrams)/total,"\n\n", "probability of birgram", len(bigrams)/total, "\n\n", "probability of trigram", len(trigrams)/total,"\n\n")
    print("Total No of Tokens :" , tokens, "\n\n")
    print("Vocabulary before processing", vocabB, "\n\n")
    print("Vocabulary after processing", vocabA, "\n\n")
    print("Average length of reviews :", avgrv/lengthofreviews,"\n\n")
    print("Average length of tokens within a review", avgtk/tokens, "\n\n")
text_mining()
