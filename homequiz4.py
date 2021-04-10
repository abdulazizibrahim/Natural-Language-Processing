'''
    Name : Abdul Aziz Muhammad Ibrahim Isa
    Roll No: p17-6143
    Natural Language Processing Home Quiz # 4
    Topic Modeling
'''
#all imports come here
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
class TopicModeling:
    def __init__(self, path):
        self.corpus = pd.read_csv(path,encoding="UTF-8", engine='c')
        self.review = []
        self.labels = []
    def ArrangeData(self):
        reviews = []
        for i in self.corpus.review:
            reviews.append(i)
        self.review = reviews
    def Preprocessing(self):
        x = self.review
        flist = []
        for i in x:
            i = i.replace('<br /><br />','')
            flist.append(i)
        self.review = flist
        #print(self.review[:2])
    def CountVect(self):
        vec = CountVectorizer()
        self.matrix = vec.fit_transform(self.review)
        self.features = vec.get_feature_names()
    def LDAModeling(self):
        inp = [10,15, 20]
        for i in inp:
            print("\n\n", ' no of topics are ', i , '\n\n')
            lda = LatentDirichletAllocation(n_components = i)
            lda.fit(self.matrix)
            for tid, topic in enumerate(lda.components_ ):
                print('topic :', tid + 1)
                print('WordID : ', topic.argsort()[:-15:-1])
                print('words : ', [self.features[i] for i in topic.argsort()[:-15:-1]])
    def main(self):
        self.ArrangeData()
        self.Preprocessing()
        self.CountVect()
        self.LDAModeling()
if __name__ == "__main__":
    obj = TopicModeling('C:\\Users\\abdul\\Downloads\\imdb-dataset-of-50k-movie-reviews\\IMDB Dataset.csv')
    obj.main()
