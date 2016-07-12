from __future__ import print_function
import nltk
import sys
from nltk.corpus import brown
import random

tagged_sents=brown.tagged_sents()
size=int(len(tagged_sents)*0.1)

train_sents=tagged_sents[size:]
brown_tags_words=[]
brown_tags=[]
brown_words=[]
for sent in train_sents:
    brown_tags_words.append(("<s>","<s>"))
    brown_tags_words.extend([(tag[:2], word) for (word, tag) in sent])
    brown_tags_words.append(("</s>","</s>"))

    brown_tags.append("<s>")
    brown_tags.extend([tag[:2] for (word, tag) in sent])
    brown_tags.append("</s>")

    brown_words.extend([word for (word, tag) in sent])

print ('a')
tag_set=set(brown_tags)
word_set=set(brown_words)
estimator = lambda fd, bins: nltk.LidstoneProbDist(fd, 0.1, bins)
cfd_tags_words=nltk.ConditionalProbDist(nltk.ConditionalFreqDist(brown_tags_words), estimator, len(word_set))	#save the data as pickle

cfd_tags=nltk.ConditionalProbDist(nltk.ConditionalFreqDist(nltk.bigrams(brown_tags)), estimator, len(tag_set))


print ('b')
print('c')
test_sents=tagged_sents[:size]
test_sent_x=test_sents[9] #in test_sents[4] error occurs #just use one sentence of test set. please change the num
test_words_x=[]
test_words_x.extend(word for (word, tag) in test_sent_x)
print ('TEST_TEXT:',end="") #we are going to tag this sentence
for x in test_words_x : print (x,end=" ") 
print ("\n")
test_answer_x=[]
test_answer_x.append("<s>")
test_answer_x.extend([tag[:2] for (word, tag) in test_sent_x])
test_answer_x.append("</s>")
print ('ANSWER:',end="") #correct tag list
for x in test_answer_x : print (x, end=" ")
print ("\n")


viterbi_list=[]
backvec_list=[]
viterbi_0={}
backvec_0={}

for i in tag_set:
	if not i=='<s>':
		viterbi_0[i]=cfd_tags['<s>'].prob(i)*cfd_tags_words[i].prob(test_words_x[0])
		backvec_0[i]='<s>'

viterbi_list.append(viterbi_0)
backvec_list.append(backvec_0)

current_best = max(viterbi_0.keys(), key = lambda tag: viterbi_0[ tag ])
print( "'" + test_words_x[0] + "'", backvec_0[current_best], current_best)

for i in range(1,len(test_words_x)): #index of test_words_x
	viterbi_x={}
	backvec_x={}
	previous_viterbi=viterbi_list[-1]
	for j in tag_set:
		if not j=='<s>':
			best_viterbi='None'
			best_prob=0.0
			for k in previous_viterbi.keys():
				calculating_prob=previous_viterbi[k]*cfd_tags[k].prob(j)*cfd_tags_words[j].prob(test_words_x[i])
				if calculating_prob>best_prob:#MLE
					best_viterbi=k
					best_prob=calculating_prob
			viterbi_x[j]=best_prob
			backvec_x[j]=best_viterbi
	current_best = max(viterbi_x.keys(), key = lambda tag: viterbi_x[tag])
	print("'" + test_words_x[i] + "'", backvec_x[current_best], current_best)
	viterbi_list.append(viterbi_x)
	backvec_list.append(backvec_x)
previous_viterbi=viterbi_list[-1]
best_viterbi='None'
best_prob=0.0
for k in previous_viterbi.keys():
	calculating_prob=previous_viterbi[k]*cfd_tags[k].prob('</s>')
	if calculating_prob>best_prob:#MLE
		best_viterbi=k
		best_prob=calculating_prob
prob_result_tags=best_prob
best_result_tags=[]
backvec_list.reverse()
best_tag=best_viterbi
for x in backvec_list:#viterbi algorithm
	if not best_tag=='<s>':
		best_result_tags.append(x[best_tag])
		best_tag=x[best_tag]
best_result_tags.reverse()
best_result_tags.extend([best_viterbi,'</s>'])	
print ('RESULT:',end="")
for x in best_result_tags:print (x, end=" ")
print ("\n")
score=0.0
for x,y in zip(test_answer_x,best_result_tags):
	if x==y:
		score+=1
print ('SCORE:',(score-2)/(len(test_answer_x)-2))
print ('PROB:', prob_result_tags)

	






