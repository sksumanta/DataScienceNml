# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:03:10 2018

@author: Sumanta
"""

# Data: Top 100 movies (http://www.imdb.com/list/ls055592025/) with title, genre, and synopsis (IMDB and Wiki)
# Goal: Put 100 movies into 5 clusters based on text mining their synopses

# the __future__ module make upgraded python version compatable with older version.
# so higher version code can be executed in older version 

from __future__ import print_function  
from IPython import get_ipython
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
# mpld3 is taking matplotlib graphs and converts them into javascript.
import mpld3  # if not there install it using anaconda navigator


# Read movie titles from E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/title_list.txt file 

with open('E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/title_list.txt') as title:
    movieTitle = title.read().split('\n')

print(movieTitle , "\n\n no of movie titles \c" , len(movieTitle) )
movieTitle = movieTitle[0:100] #  taking 100 movies as last one is empty string 

moviSerialNo = [rank for rank in range(1, len(movieTitle)+1) ]
print(moviSerialNo)

# Read Genres information from E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/genres_list.txt file 

with open('E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/genres_list.txt') as genres:
    genresList = genres.read().split('\n')

print(genresList , "\n\n no of genres titles \c" , len(genresList) )
genresList = genresList[0:100] #  taking 100 movies as last one is empty string 

# Read in the synopses from wiki E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/synopses_list_wiki.txt file 
wikiSynopses=open('E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/synopses_list_wiki.txt',encoding="utf-8").read().split('\n BREAKS HERE')
print(len(wikiSynopses))
wikiSynopses = wikiSynopses[:100]
print(wikiSynopses[0])

# strips html formatting and converts to unicode
def convertHtml2Unicode(synopses):
    cleanSynopses=[]
    for element in synopses:
        element=BeautifulSoup(element,'html.parser').getText()
        #print(element)
        cleanSynopses.append(element)
    return cleanSynopses
    
wikiSynopses = convertHtml2Unicode(wikiSynopses)
print(wikiSynopses[0])

# Read in the synopses from imdb E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/synopses_list_imdb.txt file 
imdbSynopses=open('E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/data/synopses_list_imdb.txt',encoding="utf-8").read().split('\n BREAKS HERE')
print(len(imdbSynopses))
imdbSynopses = imdbSynopses[:100]
print(imdbSynopses[0])

imdbSynopses = convertHtml2Unicode(imdbSynopses)
print(imdbSynopses[0])

#Concatinate the wiki synopses with its corresponding imdb synopses
synopses=[]
for indx in range(len(imdbSynopses)):
    synopses.append(wikiSynopses[indx]+imdbSynopses[indx])
            
print(len(synopses))
print(synopses[0])

 
        
# load nltk's English stopwords as variable called 'stopwords'
# nltk.download() to install full nltk (required one time), you will get a pop up to install 
# nltk.download('stopwords') # only for stop words

# Stop Words are words which do not contain important significance to be used in Search Queries
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")  # it will return a stemmer object

"""
################## see an example how to tokenize a sentence to words ###############

sents = [sent for sent in nltk.sent_tokenize("Today (May 19, 2016) is his only daughter's wedding. Vito Corleone is the Godfather. Vito's youngest son, Michael, in a Marine Corps uniform, introduces his girlfriend, Kay Adams, to his family at the sprawling reception.")]
print("from a long sentence we got splited sentence list as below \n\n ", sents)

words=[nltk.word_tokenize(sents[wd]) for wd in range(len(sents)) ]
print("from sentence we got splited word list as below \n\n ",words)

# filter out words and remove words not containing letters
'''
filteredWords = []
for word in words:
        if re.search('[a-zA-Z]', str(word)):
            filteredWords.append(word)
print(filteredWords)
'''       
'''
# Frequency of each word is 
fequency={}
for wd in wordStrmmer:
    fequency[wd]=onlyWords.count(wd)  
print(fequency)
'''
alphaWord=[wd for wd in words if re.search('[a-zA-Z]', str(wd))]
print("from words we got only alphabetical word list as below \n\n ",alphaWord)

# By using the stemmer keep one word for a group of synonym words 
# for example   "onli" and "wedding" is stemmed to "wed" 

stemmerWord = [stemmer.stem(str(st).lower()) for st in alphaWord]
print(stemmerWord)
####################################################################################
"""

def onlyWordTokenize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    alphaWord= [wd for wd in tokens if re.search('[a-zA-Z]', str(wd))]
    alphaWord = [w for w in alphaWord if len(w)>2] # Remove the  words less than 3 chars
#I found that names--for example "Michael" or "Tom" are found in several of the movies 
# and the synopses, but the names carry no real meaning.
    rmwd=['tom', 'mr.','james','michael','john','n\'t' ,'york' ,'new york','peter']
    for wd in alphaWord:
        if wd in rmwd:
            alphaWord.remove(wd)
    return alphaWord

# Below function define a tokenizer and stemmer which returns the set of stems 
# in the text that it is passed  Punkt Sentence Tokenizer, sent means sentence 

def tokenizeNstem(text):
    alphaWord = onlyWordTokenize(text)
    stems = [stemmer.stem(t) for t in alphaWord]
    #print(stems)
    return stems


'''
textTest = "Today (May 19, 2016) is his only daughter's wedding. Vito Corleone is the Godfather. Vito's youngest son, Michael, in a Marine Corps uniform, introduces his girlfriend, Kay Adams, to his family at the sprawling reception."
onlyWords = onlyWordTokenize(textTest)
print(onlyWords)
print(len(onlyWords))
wordStrmmer = tokenizeNstem(textTest)
print(len(wordStrmmer))

wordsDataframe = pd.DataFrame({'WORD': onlyWords}, index = wordStrmmer)
print('there are ' + str(wordsDataframe.shape[0]) + ' items in wordsDataframe')
print(wordsDataframe)

'''

totalTokens = []
totalWords = []
for wd in synopses:
    totalTokens.extend(tokenizeNstem(wd))
    totalWords.extend(onlyWordTokenize(wd))
#print(totalTokens)
print(len(totalTokens))
#print(totalWords)
print(len(totalWords))

vocabDF = pd.DataFrame({'words': totalWords}, index = totalTokens)
print('there are ' + str(vocabDF.shape[0]) + ' items in vocab data frame')
print(vocabDF.head())


# Generate TF-IDF matrix (see http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
# 
# max_df: When building the vocabulary ignore terms/words that have a document frequency strictly higher than the given threshold.
# Here I consider if the term is greater than 80% of the documents it probably cares little meanining (in the context of film synopses)
# 
# min_idf: When building the vocabulary ignore the terms/words that have a document frequency strictly lower than the given threshold (e.g. 0.5).
# Here I consider 0.2; the term/word must be in at least 20% of the document. 

#I found that names--for example "Michael" or "Tom" are found in several of the movies 
# and the synopses, but the names carry no real meaning.
#

# ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
# All values of n such that min_n <= n <= max_n will be used.
# for ngram_range, here I'll look at unigrams, bigrams and trigrams


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidfVectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenizeNstem, ngram_range=(1,3))

get_ipython().magic(u'time tfidfMatrix = tfidfVectorizer.fit_transform(synopses) #fit the vectorizer to synopses')

print(tfidfMatrix.shape)  # the matrix has 100 rows and 580 columns
terms = tfidfVectorizer.get_feature_names()
print(len(terms))


'''

from sklearn.metrics.pairwise import cosine_similarity
# A short example using the sentences above
words_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenizeNstem, ngram_range=(1,3))

get_ipython().magic(u'time words_matrix = words_vectorizer.fit_transform(sents) #fit the vectorizer to synopses')

# (3, 58) means the matrix has 3 rows (two sentences) and 58 columns (58 terms)
print(words_matrix.shape)
print(words_matrix)

# this is how we get the 58 terms
analyze = words_vectorizer.build_analyzer()
print(analyze("Today (May 19, 2016) is his only daughter's wedding."))
print(analyze("Vito Corleone is the Godfather."))
print(analyze("Vito's youngest son, Michael, in a Marine Corps uniform, introduces his girlfriend, Kay Adams, to his family at the sprawling reception."))
all_terms = words_vectorizer.get_feature_names()
print(all_terms)
print(len(all_terms))

# sent 1 and 2, similarity 0, sent 1 and 3 shares "his", sent 2 and 3 shares Vito - try to change Vito's in sent3 to His and see the similary matrix changes
example_similarity = cosine_similarity(words_matrix)
example_similarity

'''

# Now using the tf-idf matrix, you can run a slew of clustering algorithms to better understand the hidden structure within the synopses. 

'''
For K-means initial pre-determined number of clusters has been chosen 
(initially no of chosen cluster 5). 

In K-means each observation is assigned to a cluster (cluster assignment) 
,to minimize "within" cluster sum of squares. 

Next, the mean of the clustered observations is calculated and used as the 
new cluster centroid. 

Then, observations are reassigned to clusters and centroids recalculated 
iteratively until the algorithm reaches convergence (no change in centroid) .

true_k = 3
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
y=km.fit(X)
print(km)

'''

# 
# It (the below code )took several runs for k-means algorithm to converge a global optimum 
# to reaching local optima - how to decide that the algorithm converged???

from sklearn.cluster import KMeans

iniNumOfCluster = 5

km = KMeans(n_clusters = iniNumOfCluster)

get_ipython().magic(u'time km.fit(tfidfMatrix)')  # Do the actual clustering

clusters = km.labels_.tolist() #This list will be ordered the dict you passed to your vectorizer.
print(clusters)
print(len(clusters))

'''
# once it converge (no change in centroid) , reload the model/reassign 
# use joblib.dump to pickle the model and  label it as the clusters.
# By using pickel the cluster will not change for each run so we can 
# execute the below codes instade of above codes Else every time we 
# need to change the cluster dictionary given below 

'''

'''

Path='E:/datascience/pythontut/DataScienceInPy/clusteringProgram/'
import sys
sys.path.append(Path)

from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

#joblib.dump(km,  'clusterProgram.pkl')  # execute one time to save the file

km = joblib.load('clusterProgram.pkl')
clusters = km.labels_.tolist()

'''

# create a dictionary of titles, ranks, the synopsis, the cluster assignment, and the genres

films = { 'title': movieTitle, 'rank': moviSerialNo, 'synopsis': synopses, 'cluster': clusters, 'genre': genresList }

movieDataframe = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])

print(movieDataframe.head(2))
#number of films per cluster (clusters from 0 to 4)
print(movieDataframe['cluster'].value_counts())

#for aggregation purposes groupby cluster 
groupedCluster = movieDataframe['rank'].groupby(movieDataframe['cluster']) 

#average/mean rank (1 to 100) per cluster
print(groupedCluster.mean())

'''

cluster
0    38.971429
1    55.181818
2    56.457143
3    65.333333
4    60.000000
Name: rank, dtype: float64

As i got the cluster output like this so, 
clusters 0 and 1 have the lowest rank, which indicates that they (the clusters)
, on average, contain films that were ranked as "better" on the top 100 list.

'''

# how to get the content of the cluster. 
# Lets consider 6 terms/words per cluster. 

print("The terms/words per cluster\n")

#sort cluster centers by proximity to centroid
# argsort()[:, ::-1] line converts each centroid into a sorted (descending)
centroids = km.cluster_centers_.argsort()[:, ::-1] 
#print(centroids)

for num in range(iniNumOfCluster):
    print("\n\n The cluster %d words: " % num , end='')
    #DataFrame.ix useful when dealing with mixed positional and label based hierachical indexes.
    #will support any of the inputs in .loc and .iloc also supports floating point label schemes.
    for ind in centroids[num, :6]:
        print(' %s' % vocabDF.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print("\n\n Cluster %d movies titles:" % num, end='')
    for title in movieDataframe.ix[num]['title'].values.tolist():
        print(' %s,' % title, end='\n\n')


# Multidimensional scaling  
# Convert the dist matrix into a 2-dimensional array using multidimensional scaling.
# As we're plotting points in a two-dimensional plane so converting to 2D array
# Also we can use principal component analysis

from sklearn.metrics.pairwise import cosine_similarity

similarityDistance = 1 - cosine_similarity(tfidfMatrix)
print(type(similarityDistance))
print(similarityDistance.shape)
        
import os  
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS

MDS()

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

get_ipython().magic(u'time pos = mds.fit_transform(similarityDistance)  # shape (n_components, n_samples)')

xCordinet, yCordinet = pos[:, 0], pos[:, 1]       

'''
print(pos.shape)
print(pos)
xCordinet, yCordinet = pos[:, 0], pos[:, 1]       
print(xCordinet)
print(yCordinet)
'''
# visualize the document clustering output using matplotlib and mpld3

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'killed, soldiers, war', 
                 1: 'family, love, relationship', 
                 2: 'killed, police, meet', 
                 3: 'children, attempt, captain', 
                 4: 'family, father, police'}

# Now plot the labeled observations (films, film titles) colored by cluster using matplotlib.
# ipython magic to show the matplotlib plots inline
get_ipython().magic(u'matplotlib inline')

#create data frame having result of the MDS plus the cluster numbers and titles

resultDF = pd.DataFrame(dict(xAxis=xCordinet, yAxis=yCordinet, clusterNum=clusters, title=movieTitle)) 
print(resultDF[1:3]) 

#group by cluster
clusterGroup = resultDF.groupby('clusterNum')


'''
As we don't care about our plot axes are placed in the figure canvas so 
we can use plt.subplots().

plt.subplots() is a function that returns a tuple containing a figure and 
axes object(s). You can unpack this tuple into the variables fig and ax. 

If you want to change figure-level attributes or save the figure, you can 
modify fig variable. The   the figsize and dpi keyword arguments can be 
specified when the Figure object is created. The figsize is a tuple of 
width and height of the figure in inches, and dpi is the pixel per inch.

All axes objects will be in ax. 

fig, ax  = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
    
fig.tight_layout() #automatically adjusts the positions of axes on figure canvas

##################################

fig, ax = plt.subplots(figsize=(20, 10))
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');

'''
# set up plot
fig, ax = plt.subplots(figsize=(20, 10)) 

# ax.margins(0.05) # just adds 5% padding to the autoscaling

#Now iterate through cluster groups "clusterGroup" to layer the plot
for name, groupData in clusterGroup:
    #print("      group name       " + str(name) +"       " + str(groupData ))
    #print(groupData.iloc[:,2])
    ax.plot(groupData.xAxis , groupData.yAxis, marker='*', linestyle='', ms=20, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    
    ax.tick_params( axis= 'x',        # changes apply to the x-axis
                   which='both',      # both major and minor ticks are affected
                   bottom='off',      # ticks along the bottom edge are off
                   top='off',         # ticks along the top edge are off
                   labelbottom='off')
     
    ax.tick_params( axis= 'y',        # changes apply to the y-axis
                   which='both',      # both major and minor ticks are affected
                   left='off',        # ticks along the bottom edge are off
                   top='off',         # ticks along the top edge are off
                   labelleft='off')
    
#decorate a figure with titles, axis labels, and legends.
    
ax.legend(numpoints=1)   

#add label in x,y position with the label as the film title
for i in range(len(resultDF)):
    ax.text(resultDF.ix[i]['xAxis'], resultDF.ix[i]['yAxis'], resultDF.ix[i]['title'], size=10)   

# To save the plot if need be (dpi ----- to set the pixel)

plt.savefig('E:/datascienceProject/DatascienceNml/MLProjects/clusteringProgram/Movieclusters.png', dpi=200)

#show the plot
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.show()

