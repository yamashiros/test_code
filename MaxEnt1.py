from __future__ import print_function
import nltk
import sys
import random
import glob
import pickle

pos_text_fnames = glob.glob('txt_sentoken/pos/*.txt')
size=int(len(pos_text_fnames)*0.1)
pos_train_sents=[]
pos_test_sents=[]
for x in pos_text_fnames[size:]:
	f=open(x)
	pos_train_sents.append(f.read())	
	f.close()
for x in pos_text_fnames[:size]:	
	f=open(x)
	pos_test_sents.append(f.read())	
	f.close()

neg_text_fnames = glob.glob('txt_sentoken/neg/*.txt')
size=int(len(neg_text_fnames)*0.1)
neg_train_sents=[]
neg_test_sents=[]
for x in neg_text_fnames[size:]:	
	f=open(x)
	neg_train_sents.append(f.read())	
	f.close()
for x in neg_text_fnames[:size]:	
	f=open(x)
	neg_test_sents.append(f.read())	
	f.close()

pos_freq_words_set=None
neg_freq_words_set=None
try:	#if you've made these models, read it (pickle file). 
	f1 = open ('pos_freq_words_set.pickle', 'rb')
	pos_freq_words_set=pickle.load(f1)
except  IOError as e:
	if pos_freq_words_set==None:
		freq_words=[]
		sw = set(nltk.corpus.stopwords.words("english"))

		train_list=[]
		for s in pos_train_sents:
			for t in nltk.word_tokenize(s):
				train_list.append(nltk.WordNetLemmatizer().lemmatize(t.lower()))
		pos_fdist=nltk.FreqDist(train_list)
		for w in sorted(pos_fdist):
			if not w in sw:
				freq_words.append(w) 
			if len(freq_words)>=500:
				break
		print (len(freq_words))
			 

		pos_freq_words_set=set(freq_words)
		with open('pos_freq_words_set.pickle','wb') as f:
			pickle.dump(pos_freq_words_set,f)
try:	#if you've made these models, read it (pickle file). 
	f2 = open ('neg_freq_words_set.pickle', 'rb')
	neg_freq_words_set=pickle.load(f2)
except  IOError as e:
	if neg_freq_words_set==None:
		freq_words=[]
		sw = set(nltk.corpus.stopwords.words("english"))

		train_list=[]
		for s in neg_train_sents:
			for t in nltk.word_tokenize(s):
				train_list.append(nltk.WordNetLemmatizer().lemmatize(t.lower()))
		neg_fdist=nltk.FreqDist(train_list)
		
		for w in sorted(neg_fdist):
			if not w in sw:
				freq_words.append(w) 
			if len(freq_words)>=500:
				break
		print (len(freq_words))
		neg_freq_words_set=set(freq_words)
		with open('neg_freq_words_set.pickle','wb') as f:
			pickle.dump(neg_freq_words_set,f)
freq_words_set=pos_freq_words_set^neg_freq_words_set
print (len(freq_words_set))
print (len(pos_freq_words_set&neg_freq_words_set))
print (len(pos_freq_words_set|neg_freq_words_set))
train_data=[]
for s in pos_train_sents:
	pos_dict={}
	for w in freq_words_set:
		pos_dict[w]=0
	for t in nltk.word_tokenize(s):
		w=nltk.WordNetLemmatizer().lemmatize(t.lower())
		if w in freq_words_set and pos_dict[w]==1:
			pos_dict[w]+=1
	train_data.append((pos_dict,'pos'))
			
for s in neg_train_sents:
	neg_dict={}
	for w in freq_words_set:
		neg_dict[w]=0
	for t in nltk.word_tokenize(s):
		w=nltk.WordNetLemmatizer().lemmatize(t.lower())
		if w in freq_words_set:
			neg_dict[w]+=1
	train_data.append((neg_dict,'neg'))


pos_test_data=[]
for s in pos_test_sents:
	pos_dict={}
	for w in freq_words_set:
		pos_dict[w]=0
	for t in nltk.word_tokenize(s):
		w=nltk.WordNetLemmatizer().lemmatize(t.lower())
		if w in freq_words_set:
			pos_dict[w]+=1
	pos_test_data.append((pos_dict))

neg_test_data=[]
for s in neg_test_sents:
	neg_dict={}
	for w in freq_words_set:
		neg_dict[w]=0
	for t in nltk.word_tokenize(s):
		w=nltk.WordNetLemmatizer().lemmatize(t.lower())
		if w in freq_words_set:
			neg_dict[w]+=1
	neg_test_data.append((neg_dict))

			
total_score=0.0
length=0.0

algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0] #GIS
try:
	classifier = nltk.classify.MaxentClassifier.train(train_data, algorithm, max_iter=10)
except Exception as e:
	print('Error: %r' % e)

for featureset in pos_test_data:
	pdist = classifier.prob_classify(featureset)
	if pdist.prob('pos')>pdist.prob('neg'):
		total_score+=1
	length+=1
#print ('SCORE:',total_score)
for featureset in neg_test_data:
	pdist = classifier.prob_classify(featureset)
	if pdist.prob('pos')<pdist.prob('neg'):
		total_score+=1
	length+=1

print ('TOTAL SCORE:',total_score/length)

