# NLP Home Quiz # 1
# Name : Abdul Aziz Muhammad Ibrahim Isa
# Roll no: P17- 6143

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = open('C:\\Users\\abdul\\Downloads\\Movies_TV.txt').read()
def nlp():
	vec = TfidfVectorizer(ngram_range =(1, 3), binary = 'true', min_df = 10, max_df= 100, max_features = 1000)
	data = corpus.split('\n')
	y = vec.fit_transform(data)
	print(y.toarray())
	print(vec.get_feature_names())
	print("no of features ", len(vec.get_feature_names()))
nlp()
