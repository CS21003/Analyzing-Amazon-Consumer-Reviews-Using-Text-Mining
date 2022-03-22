#--------------------------------------------------#
# Preparation #
#--------------------------------------------------#
# Import Modules
import re
import string
import sys
import csv
import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# For data cleaning
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# LDA
from gensim import corpora
from gensim.models.ldamodel import LdaModel

import pickle

from google.colab import drive
drive.mount('/content/drive')


#--------------------------------------------------#
# Topic Modelling #
#--------------------------------------------------#
punc = string.punctuation
lemm = WordNetLemmatizer()

stop = [
    "'ll", "'ve", '0o', '0s', '3a', '3b', '3d', '6b', '6o', 'a', "a's", 'a1', 'a2', 
    'a3', 'a4', 'ab', 'able', 'about', 'above', 'abst', 'ac', 'accordance', 'according', 
    'accordingly', 'across', 'act', 'actually', 'ad', 'added', 'adj', 'ae', 'af', 'affected', 
    'affecting', 'affects', 'after', 'afterwards', 'ag', 'again', 'against', 'ah', 'ain', "ain't", 
    'aj', 'al', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 
    'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'announce', 'another', 'any', 
    'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'ao', 'ap', 
    'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'ar', 'are', 'aren', 
    "aren't", 'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'au', 'auth', 
    'av', 'available', 'aw', 'away', 'awfully', 'ax', 'ay', 'az', 'b', 'b1', 'b2', 'b3', 'ba', 'back', 'bc', 
    'bd', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin', 
    'beginning', 'beginnings', 'begins', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 
    'better', 'between', 'beyond', 'bi', 'bill', 'biol', 'bj', 'bk', 'bl', 'bn', 'both', 'bottom', 'bp', 'br', 
    'brief', 'briefly', 'bs', 'bt', 'bu', 'but', 'bx', 'by', 'c', "c'mon", "c's", 'c1', 'c2', 'c3', 'ca', 'call', 
    'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'cc', 'cd', 'ce', 'certain', 'certainly', 'cf', 
    'cg', 'ch', 'changes', 'ci', 'cit', 'cj', 'cl', 'clearly', 'cm', 'cn', 'co', 'com', 'come', 'comes', 'con', 
    'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 
    'could', 'couldn', "couldn't", 'couldnt', 'course', 'cp', 'cq', 'cr', 'cry', 'cs', 'ct', 'cu', 'currently', 
    'cv', 'cx', 'cy', 'cz', 'd', 'd2', 'da', 'date', 'dc', 'dd', 'de', 'definitely', 'describe', 'described', 
    'despite', 'detail', 'df', 'di', 'did', 'didn', "didn't", 'different', 'dj', 'dk', 'dl', 'do', 'does', 'doesn', 
    "doesn't", 'doing', 'don', "don't", 'done', 'down', 'downwards', 'dp', 'dr', 'ds', 'dt', 'du', 'due', 'during', 
    'dx', 'dy', 'e', 'e2', 'e3', 'ea', 'each', 'ec', 'ed', 'edu', 'ee', 'ef', 'effect', 'eg', 'ei', 'eight', 'eighty', 
    'either', 'ej', 'el', 'eleven', 'else', 'elsewhere', 'em', 'empty', 'en', 'end', 'ending', 'enough', 'entirely', 
    'eo', 'ep', 'eq', 'er', 'es', 'especially', 'est', 'et', 'et-al', 'etc', 'eu', 'ev', 'even', 'ever', 'every', 
    'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'ey', 'f', 'f2', 'fa', 
    'far', 'fc', 'few', 'ff', 'fi', 'fifteen', 'fifth', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'fix', 'fj', 
    'fl', 'fn', 'fo', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'forty', 'found', 'four', 
    'fr', 'from', 'front', 'fs', 'ft', 'fu', 'full', 'further', 'furthermore', 'fy', 'g', 'ga', 'gave', 'ge', 'get', 'gets', 
    'getting', 'gi', 'give', 'given', 'gives', 'giving', 'gj', 'gl', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'gr', 
    'greetings', 'gs', 'gy', 'h', 'h2', 'h3', 'had', 'hadn', "hadn't", 'happens', 'hardly', 'has', 'hasn', "hasn't", 'hasnt', 
    'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hed', 'hello', 'help', 'hence', 'her', 'here', 
    "here's", 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes', 'hh', 'hi', 'hid', 'him', 
    'himself', 'his', 'hither', 'hj', 'ho', 'home', 'hopefully', 'how', "how's", 'howbeit', 'however', 'hr', 'hs', 'http', 
    'hu', 'hundred', 'hy', 'i', "i'd", "i'll", "i'm", "i've", 'i2', 'i3', 'i4', 'i6', 'i7', 'i8', 'ia', 'ib', 'ibid', 'ic', 
    'id', 'ie', 'if', 'ig', 'ignored', 'ih', 'ii', 'ij', 'il', 'im', 'immediate', 'immediately', 'importance', 'important', 
    'in', 'inasmuch', 'inc', 'indeed', 'index', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'insofar', 
    'instead', 'interest', 'into', 'invention', 'inward', 'io', 'ip', 'iq', 'ir', 'is', 'isn', "isn't", 'it', "it'd", "it'll", 
    "it's", 'itd', 'its', 'itself', 'iv', 'ix', 'iy', 'iz', 'j', 'jj', 'jr', 'js', 'jt', 'ju', 'just', 'k', 'ke', 'keep', 
    'keeps', 'kept', 'kg', 'kj', 'km', 'know', 'known', 'knows', 'ko', 'l', 'l2', 'la', 'largely', 'last', 'lately', 'later', 
    'latter', 'latterly', 'lb', 'lc', 'le', 'least', 'les', 'less', 'lest', 'let', "let's", 'lets', 'lf', 'like', 'liked', 
    'likely', 'line', 'little', 'lj', 'll', 'ln', 'lo', 'look', 'looking', 'looks', 'los', 'lr', 'ls', 'lt', 'ltd', 'm', 'm2', 
    'ma', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 
    'mg', 'might', 'mightn', "mightn't", 'mill', 'million', 'mine', 'miss', 'ml', 'mn', 'mo', 'more', 'moreover', 'most', 'mostly', 
    'move', 'mr', 'mrs', 'ms', 'mt', 'mu', 'much', 'mug', 'must', 'mustn', "mustn't", 'my', 'myself', 'n', 'n2', 'na', 'name', 
    'namely', 'nay', 'nc', 'nd', 'ne', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needn', "needn't", 'needs', 'neither', 
    'never', 'nevertheless', 'new', 'next', 'ng', 'ni', 'nine', 'ninety', 'nj', 'nl', 'nn', 'no', 'nobody', 'non', 'none', 'nonetheless', 
    'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'novel', 'now', 'nowhere', 'nr', 'ns', 'nt', 'ny', 'o', 'oa', 'ob', 
    'obtain', 'obtained', 'obviously', 'oc', 'od', 'of', 'off', 'often', 'og', 'oh', 'oi', 'oj', 'ok', 'okay', 'ol', 'old', 'om', 
    'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'oo', 'op', 'oq', 'or', 'ord', 'os', 'ot', 'other', 'others', 'otherwise', 
    'ou', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'ow', 'owing', 'own', 'ox', 'oz', 'p', 'p1', 'p2', 
    'p3', 'page', 'pagecount', 'pages', 'par', 'part', 'particular', 'particularly', 'pas', 'past', 'pc', 'pd', 'pe', 'per', 'perhaps', 
    'pf', 'ph', 'pi', 'pj', 'pk', 'pl', 'placed', 'please', 'plus', 'pm', 'pn', 'po', 'poorly', 'possible', 'possibly', 'potentially', 
    'pp', 'pq', 'pr', 'predominantly', 'present', 'presumably', 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 
    'ps', 'pt', 'pu', 'put', 'py', 'q', 'qj', 'qu', 'que', 'quickly', 'quite', 'qv', 'r', 'r2', 'ra', 'ran', 'rather', 'rc', 'rd', 're', 
    'readily', 'really', 'reasonably', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 
    'research', 'research-articl', 'respectively', 'resulted', 'resulting', 'results', 'rf', 'rh', 'ri', 'right', 'rj', 'rl', 'rm', 'rn', 
    'ro', 'rq', 'rr', 'rs', 'rt', 'ru', 'run', 'rv', 'ry', 's', 's2', 'sa', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sc', 'sd', 
    'se', 'sec', 'second', 'secondly', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 
    'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'sf', 'shall', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 
    'shed', 'shes', 'should', "should've", 'shouldn', "shouldn't", 'show', 'showed', 'shown', 'showns', 'shows', 'si', 'side', 'significant', 
    'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'sixty', 'sj', 'sl', 'slightly', 'sm', 'sn', 'so', 'some', 'somebody', 
    'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'sp', 'specifically', 
    'specified', 'specify', 'specifying', 'sq', 'sr', 'ss', 'st', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such', 
    'sufficiently', 'suggest', 'sup', 'sure', 'sy', 'system', 'sz', 't', "t's", 't1', 't2', 't3', 'take', 'taken', 'taking', 'tb', 'tc', 'td', 
    'te', 'tell', 'ten', 'tends', 'tf', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've", 'thats', 'the', 
    'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'll", "there's", "there've", 'thereafter', 'thereby', 'thered', 
    'therefore', 'therein', 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 
    'theyd', 'theyre', 'thickv', 'thin', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh', 'thousand', 
    'three', 'throug', 'through', 'throughout', 'thru', 'thus', 'ti', 'til', 'tip', 'tj', 'tl', 'tm', 'tn', 'to', 'together', 'too', 'took', 
    'top', 'toward', 'towards', 'tp', 'tq', 'tr', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'tt', 'tv', 'twelve', 'twenty', 'twice', 
    'two', 'tx', 'u', 'u201d', 'ue', 'ui', 'uj', 'uk', 'um', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 
    'uo', 'up', 'upon', 'ups', 'ur', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'ut', 'v', 'va', 
    'value', 'various', 'vd', 've', 'very', 'via', 'viz', 'vj', 'vo', 'vol', 'vols', 'volumtype', 'vq', 'vs', 'vt', 'vu', 'w', 'wa', 'want', 
    'wants', 'was', 'wasn', "wasn't", 'wasnt', 'way', 'we', "we'd", "we'll", "we're", "we've", 'wed', 'welcome', 'well', 'well-b', 'went', 
    'were', 'weren', "weren't", 'werent', 'what', "what'll", "what's", 'whatever', 'whats', 'when', "when's", 'whence', 'whenever', 'where', 
    "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whim', 'whither', 
    'who', "who'll", "who's", 'whod', 'whoever', 'whole', 'whom', 'whomever', 'whos', 'whose', 'why', "why's", 'wi', 'widely', 'will', 'willing', 
    'wish', 'with', 'within', 'without', 'wo', 'won', "won't", 'wonder', 'wont', 'words', 'world', 'would', 'wouldn', "wouldn't", 'wouldnt', 
    'www', 'x', 'x1', 'x2', 'x3', 'xf', 'xi', 'xj', 'xk', 'xl', 'xn', 'xo', 'xs', 'xt', 'xv', 'xx', 'y', 'y2', 'yes', 'yet', 'yj', 'yl', 'you', 
    "you'd", "you'll", "you're", "you've", 'youd', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'yr', 'ys', 'yt', 'z', 'zero', 'zi', 'zz']

stop += ['amp']

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt * .9999)

def tokenize(line):
    line = re.sub('https?://\S+|[^\x00-\x7F]+|[0-9]+|\s+|#(\w+)', ' ', line)
    line = line.translate(str.maketrans({k: ' ' for k in punc}))
    word = [w for w in re.split(' +', line.lower()) if not w in stop and len(w) > 0]

    token = []
    for w, t in pos_tag(word):

        if t[0].lower() == 'j':
            token.append(lemm.lemmatize(w, pos = 'a'))

        elif t[0].lower() in ['v', 'n', 'r']:
            token.append(lemm.lemmatize(w, pos = t[0].lower()))

        else:
            token.append(lemm.lemmatize(w))

    return token

TEXT = pd.read_csv('/content/drive/My Drive/csv_files/01_gro_rtext.csv')
TEXT.head()

TEXT['text'] = TEXT['text'].apply(lambda x: tokenize(x) if type(x) == str else x)
TEXT = TEXT.dropna()

def clean_text(x):
  if type(x) != str:
    return x
  
  x_lst = x.split()
  if (len(x_lst) < 5 or len(x_lst) > 100):
    return np.nan
  else:
    return x

TEXT['text'] = TEXT['text'].apply(lambda x: clean_text(x))
TEXT = TEXT.dropna()

TEXT.to_csv('/content/drive/My Drive/csv_files/gro_TEXT.csv')
TEXT = pd.read_csv('/content/drive/My Drive/csv_files/gro_TEXT.csv')

TEXT = TEXT.drop(columns = 'Unnamed: 0')
TEXT

text = TEXT['text']
text = [text.replace('[', '') for text in text]
text = [text.replace(']', '') for text in text]
text = [text.replace('\'', '') for text in text]
text = [text.replace(' ', '') for text in text]
text = [text.split(',') for text in text]

id2word = corpora.Dictionary(text)
id2word.filter_extremes(no_below = 10, no_above = .5)
dtm = [id2word.doc2bow(t) for t in text]

pickle.dump(text, open('/content/drive/My Drive/csv_files/01_gro_rtext_text.pk', 'wb'))
pickle.dump(dtm, open('/content/drive/My Drive/csv_files/01_gro_rtext_dtm.pk', 'wb'))
id2word.save('/content/drive/My Drive/csv_files/01_gro_rtext_id2word')

text = pickle.load(open('/content/drive/My Drive/csv_files/01_gro_rtext_text.pk', 'rb'))
dtm = pickle.load(open('/content/drive/My Drive/csv_files/01_gro_rtext_dtm.pk', 'rb'))
id2word = corpora.Dictionary.load('/content/drive/My Drive/csv_files/01_gro_rtext_id2word')

# Run LDA model on the document term matrix
num_topics = 5
model = LdaModel(corpus = dtm,
                 id2word = id2word,
                 num_topics = num_topics,
                 random_state = 100,
                 update_every = 1,
                 chunksize = 100,
                 passes = 10,
                 alpha = 'asymmetric',
                 per_word_topics = True)

model.save('/content/drive/My Drive/csv_files/gro_model.pk')
model = LdaModel.load('/content/drive/My Drive/csv_files/gro_model.pk')

def plot_top_words(lda=model, nb_topics=5, nb_words=30):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = mpl.gridspec.GridSpec(round(math.sqrt(5))+1,round(math.sqrt(5))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(30,30))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i+1))
        
plot_top_words()

def plot_top_words(lda=model, nb_topics=5, nb_words=50):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = mpl.gridspec.GridSpec(round(math.sqrt(5))+1,round(math.sqrt(5))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(50,50))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i+1))
        
plot_top_words()


#--------------------------------------------------#
# Tables #
#--------------------------------------------------#
text = pickle.load(open('/content/drive/My Drive/csv_files/01_gro_rtext_text.pk', 'rb'))
dtm = pickle.load(open('/content/drive/My Drive/csv_files/01_gro_rtext_dtm.pk', 'rb'))
id2word = corpora.Dictionary.load('/content/drive/My Drive/csv_files/01_gro_rtext_id2word')

word_table = []

for wordID in id2word.iterkeys():
  result = model.get_term_topics(wordID, minimum_probability=0.000001)
  if (len(result) == 0):
    continue

  result = result[0]
  word = id2word.get(wordID)
  topicID = result[0] + 1
  topicProb = result[1]
  word_table.append((word, topicID, topicProb))

word_table = pd.DataFrame(word_table, columns = ['Word', 'Topic', 'Probability'])
word_table

review_table = []

def createReviewProbs(review, probs):
  if len(probs) == 5:
    probs = [topicProbs[1] for topicProbs in probs]
    review.extend(probs)
  else:
    topicProbs = [0,0,0,0,0]
    for topic in probs:
      topicProbs[topic[0]] = topic[1]
    review.extend(topicProbs)
  return review

i = 1

for _, (asin, user, text) in TEXT.iterrows():
  text = text.replace('[', '')
  text = text.replace(']', '')
  text = text.replace('\'', '')
  text = text.replace(' ', '')
  text = text.split(',')
  bow = id2word.doc2bow(text)
  result = model.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
  review = [asin, user]
  review = createReviewProbs(review, result)
  i += 1
  review = (asin, user, result[0][1], result[1][1], result[2][1], result[3][1], result[4][1])
  review_table.append(review)

review_table = pd.DataFrame(review_table, columns = ['asin', 'user', 'text', 'unknown', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5'])
review_table = review_table.drop(columns = 'Topic 4')
review_table = review_table.drop(columns = 'Topic 5')
review_table = review_table.rename(columns = {'Topic 3' : 'Topic 5'})
review_table = review_table.rename(columns = {'Topic 2' : 'Topic 4'})
review_table = review_table.rename(columns = {'Topic 1' : 'Topic 3'})
review_table = review_table.rename(columns = {'unknown' : 'Topic 2'})
review_table = review_table.rename(columns = {'text' : 'Topic 1'})

review_table = pd.DataFrame(review_table, columns = ['asin', 'user', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5'])
review_table

review_table.to_csv('/content/drive/My Drive/csv_files/gro_reviewtable.csv')

dummy_rtable = review_table
dummy_rtable = pd.read_csv('/content/drive/My Drive/csv_files/gro_reviewtable.csv')

dummy = dummy_rtable[['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']]
m = np.zeros_like(dummy.values)
m[np.arange(len(dummy)), dummy.values.argmax(1)] = 1

dummy = pd.DataFrame(m, columns = dummy.columns).astype(int)
dummy[['asin', 'user']] = dummy_rtable[['asin', 'user']]
dummy = dummy[['asin', 'user', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']]

dummy_rtable = dummy
dummy_rtable

dummy_rtable.to_csv('/content/drive/My Drive/csv_files/gro_dummy_rtable.csv')
dummy_rtable = pd.read_csv('/content/drive/My Drive/csv_files/gro_dummy_rtable.csv')

def maxDummify(x):
  maxIndex = np.argmax(x[['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']])
  dummy = [0, 0, 0, 0, 0]
  dummy[maxIndex] = 1
  x[['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']] = dummy
  return x

dummy_rtable = dummy_rtable.apply(maxDummify, 1)
dummy_rtable = dummy_rtable.drop(columns = 'Unnamed: 0')
dummy_rtable

dummy_rtable.to_csv('/content/drive/My Drive/csv_files/gro_dummy_rtable.csv')


#--------------------------------------------------#
# Regression #
#--------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data_rinfo = pd.read_csv('/content/drive/My Drive/csv_files/01_gro_rinfo.csv')

data_rinfo.replace({False: 0, True: 1}, inplace=True)
data_rinfo["helpful"] = np.where(data_rinfo["vote"]== 0, 0, 1)
 
reviewers = data_rinfo.groupby('asin')['user'].count()
data_rinfo = pd.merge(data_rinfo, reviewers, on = 'asin')
data_rinfo = data_rinfo.rename(columns = {'user_x' : 'user'})
data_rinfo = data_rinfo.rename(columns = {'user_y' : 'userCount'})

avgRating = pd.DataFrame(data_rinfo.groupby('asin')['rating'].mean())
data_rinfo = pd.merge(data_rinfo, avgRating, on = 'asin')
data_rinfo = data_rinfo.rename(columns = {'rating_x' : 'revRating'})
data_rinfo = data_rinfo.rename(columns = {'rating_y' : 'pdtAvgRating'})

data_rinfo

data_user = pd.read_csv('/content/drive/My Drive/csv_files/01_gro_user.csv')

import time
from datetime import date

data_user["userRep"] = data_user["vote"]/data_user["review"]
data_user['first'] = (pd.to_datetime(data_user['first'], format = '%Y-%m-%d')) 
data_user = data_user.rename(columns = {'first' : 'firstPost'})
data_user["timeElapsed"] = data_user["firstPost"].apply(lambda x: (date(2018, 10, 31) - x.date()))

data_user

data_meta = pd.read_csv('/content/drive/My Drive/csv_files/01_gro_meta.csv')

data_meta['titleWordCount'] = data_meta['title'].str.split()
data_meta['titleWordCount'] = data_meta['titleWordCount'].str.len()

data_meta['descWordCount'] = data_meta['desc'].str.split()
data_meta['descWordCount'] = data_meta['descWordCount'].str.len()

data_meta

data = pd.merge(data_rinfo, data_user, on = 'user')

data = data.rename(columns = {'ver_x' : 'verified'})
data = data.rename(columns = {'ver_y' : 'ver_written'})
data = data.rename(columns = {'vote_x' : 'votesPerReview'})
data = data.rename(columns = {'vote_y' : 'votesPerUser'})
data = data.rename(columns = {'avgr' : 'userAvgRating'})
data = data.rename(columns = {'wcount' : 'rtextCount'})
data = data.rename(columns = {'arrive' : 'revOrder'})
data = data.rename(columns = {'review' : 'revPerUser'})
data = data.rename(columns = {'help' : 'helpRevPerUser'})

data

data = pd.merge(data, data_meta, on = 'asin')

data = data.rename(columns = {'under' : 'subCat'})
data = data.rename(columns = {'review' : 'revPerPdt'})
data = data.rename(columns = {'help' : 'helpRevPerPdt'})
data = data.rename(columns = {'vtotal' : 'helpVotesPerPdt'})

data

review_table = pd.read_csv('/content/drive/My Drive/csv_files/gro_dummy_rtable.csv')
data = pd.merge(data, review_table, on = ['asin','user'])
data = data.dropna()
data = data.drop(columns = 'Unnamed: 0')
data

# Drop variables
data = data.drop(columns = 'votesPerReview')
data = data.drop(columns = 'userCount')
data = data.drop(columns = 'firstPost')
data = data.drop(columns = 'votesPerUser')
data = data.drop(columns = 'helpRevPerUser')
data = data.drop(columns = 'timeElapsed')
data = data.drop(columns = 'title')
data = data.drop(columns = 'subCat')
data = data.drop(columns = 'desc')
data = data.drop(columns = 'revPerPdt')
data = data.drop(columns = 'helpRevPerPdt')
data = data.drop(columns = 'helpVotesPerPdt')
data = data.drop(columns = 'Topic 5')

# Rename variables
data = data.rename(columns = {'helpful' : 'HELPFUL'})
data = data.rename(columns = {'verified' : 'VerifiedReview'})
data = data.rename(columns = {'revRating' : 'ReviewRating'})
data = data.rename(columns = {'rtextCount' : 'ReviewLength'})
data = data.rename(columns = {'image' : 'ReviewImages'})
data = data.rename(columns = {'revOrder' : 'ReviewPostingOrder'})
data = data.rename(columns = {'revPerUser' : 'ReviewAuthorExperience'})
data = data.rename(columns = {'userRep' : 'ReviewAuthorReputation'})
data = data.rename(columns = {'ver_written' : 'ReviewAuthorCredibility'})
data = data.rename(columns = {'userAvgRating' : 'AverageRatingByReviewAuthor'})
data = data.rename(columns = {'pdtAvgRating' : 'ProductAverageRating'})
data = data.rename(columns = {'rank' : 'ProductRank'})
data = data.rename(columns = {'price' : 'ProductPrice'})
data = data.rename(columns = {'titleWordCount' : 'ProductTitleLength'})
data = data.rename(columns = {'descWordCount' : 'ProductDescriptionLength'})
data = data.rename(columns = {'Topic 1' : '"Beverages"'})
data = data.rename(columns = {'Topic 2' : '"Confectionery"'})
data = data.rename(columns = {'Topic 3' : '"Flavor"'})
data = data.rename(columns = {'Topic 4' : '"ProductAppearance"'})

# Order columns
data = data[['asin', 'user', 'HELPFUL', 'VerifiedReview', 'ReviewRating', 'ReviewLength', 
             'ReviewImages', 'ReviewPostingOrder', 'ReviewAuthorExperience', 
             'ReviewAuthorReputation', 'ReviewAuthorCredibility', 'AverageRatingByReviewAuthor', 
             'ProductAverageRating', 'ProductRank', 'ProductPrice', 'ProductTitleLength', 'ProductDescriptionLength', 
             '"Beverages"', '"Confectionery"', '"Flavor"', '"ProductAppearance"']]

data

data.to_csv('/content/drive/My Drive/csv_files/gro_data.csv')
data = pd.read_csv('/content/drive/My Drive/csv_files/gro_data.csv') 
data.nunique()

x = data['helpful'].value_counts()
print(x)

df1 = pddf1 = pd.DataFrame({'ReviewRating' : data['ReviewRating'], 'reviewCount': data.groupby('ReviewRating')['ReviewRating'].transform('count')})
df2 = pd.DataFrame({'ReviewLength' : data['ReviewLength'], 'reviewCount': data.groupby('ReviewLength')['ReviewLength'].transform('count')})
df3 = pd.DataFrame({'ReviewImages' : data['ReviewImages'], 'reviewCount': data.groupby('ReviewImages')['ReviewImages'].transform('count')})
df4 = pd.DataFrame({'ReviewPostingOrder' : data['ReviewPostingOrder'], 'reviewCount': data.groupby('ReviewPostingOrder')['ReviewPostingOrder'].transform('count')})
df5 = pd.DataFrame({'ReviewAuthorExperience' : data['ReviewAuthorExperience'], 'reviewCount': data.groupby('ReviewAuthorExperience')['ReviewAuthorExperience'].transform('count')})
df6 = pd.DataFrame({'ReviewAuthorReputation' : data['ReviewAuthorReputation'], 'reviewCount': data.groupby('ReviewAuthorReputation')['ReviewAuthorReputation'].transform('count')})
df7 = pd.DataFrame({'ReviewAuthorCredibility' : data['ReviewAuthorCredibility'], 'reviewCount': data.groupby('ReviewAuthorCredibility')['ReviewAuthorCredibility'].transform('count')})
df8 = pd.DataFrame({'AverageRatingByReviewAuthor' : data['AverageRatingByReviewAuthor'], 'reviewCount': data.groupby('AverageRatingByReviewAuthor')['AverageRatingByReviewAuthor'].transform('count')})
df9 = pd.DataFrame({'ProductAverageRating' : data['ProductAverageRating'], 'reviewCount': data.groupby('ProductAverageRating')['ProductAverageRating'].transform('count')})
df10 = pd.DataFrame({'ProductRank' : data['ProductRank'], 'reviewCount': data.groupby('ProductRank')['ProductRank'].transform('count')})
df11 = pd.DataFrame({'ProductPrice' : data['ProductPrice'], 'reviewCount': data.groupby('ProductPrice')['ProductPrice'].transform('count')})
df12 = pd.DataFrame({'ProductTitleLength' : data['ProductTitleLength'], 'reviewCount': data.groupby('ProductTitleLength')['ProductTitleLength'].transform('count')})
df13 = pd.DataFrame({'ProductDescriptionLength' : data['ProductDescriptionLength'], 'reviewCount': data.groupby('ProductDescriptionLength')['ProductDescriptionLength'].transform('count')})

np.random.seed(0)
data_log = data[['HELPFUL']].copy()
data_log['VerifiedReview'] = data['VerifiedReview']

df1['ReviewRating'] = df1['ReviewRating'] - df1['ReviewRating'].min() + 1
data_log['ReviewRating'] = np.log(df1['ReviewRating'])

df2['ReviewLength'] = df2['ReviewLength'] - df2['ReviewLength'].min() + 1
data_log['ReviewLength'] = np.log(df2['ReviewLength'])

df3['ReviewImages'] = df3['ReviewImages'] - df3['ReviewImages'].min() + 1
data_log['ReviewImages'] = np.log(df3['ReviewImages'])

df4['ReviewPostingOrder'] = df4['ReviewPostingOrder'] - df4['ReviewPostingOrder'].min() + 1
data_log['ReviewPostingOrder'] = np.log(df4['ReviewPostingOrder'])

df5['ReviewAuthorExperience'] = df5['ReviewAuthorExperience'] - df5['ReviewAuthorExperience'].min() + 1
data_log['ReviewAuthorExperience'] = np.log(df5['ReviewAuthorExperience'])

df6['ReviewAuthorReputation'] = df6['ReviewAuthorReputation'] - df6['ReviewAuthorReputation'].min() + 1
data_log['ReviewAuthorReputation'] = np.log(df6['ReviewAuthorReputation'])

df7['ReviewAuthorCredibility'] = df7['ReviewAuthorCredibility'] - df7['ReviewAuthorCredibility'].min() + 1
data_log['ReviewAuthorCredibility'] = np.log(df7['ReviewAuthorCredibility'])

df8['AverageRatingByReviewAuthor'] = df8['AverageRatingByReviewAuthor'] - df8['AverageRatingByReviewAuthor'].min() + 1
data_log['AverageRatingByReviewAuthor'] = np.log(df8['AverageRatingByReviewAuthor'])

df9['ProductAverageRating'] = df9['ProductAverageRating'] - df9['ProductAverageRating'].min() + 1
data_log['ProductAverageRating'] = np.log(df9['ProductAverageRating'])

df10['ProductRank'] = df10['ProductRank'] - df10['ProductRank'].min() + 1
data_log['ProductRank'] = np.log(df10['ProductRank'])

df11['ProductPrice'] = df11['ProductPrice'] - df11['ProductPrice'].min() + 1
data_log['ProductPrice'] = np.log(df11['ProductPrice'])

df12['ProductTitleLength'] = df12['ProductTitleLength'] - df12['ProductTitleLength'].min() + 1
data_log['ProductTitleLength'] = np.log(df12['ProductTitleLength'])

df13['ProductDescriptionLength'] = df13['ProductDescriptionLength'] - df13['ProductDescriptionLength'].min() + 1
data_log['ProductDescriptionLength'] = np.log(df13['ProductDescriptionLength'])

data_log['"Beverages"'] = data['"Beverages"']
data_log['"Confectionery"'] = data['"Confectionery"']
data_log['"Flavor"'] = data['"Flavor"']
data_log['"ProductAppearance"'] = data['"ProductAppearance"']

data_log

data_log.to_csv('/content/drive/My Drive/csv_files/gro_data_log.csv')
data_log = pd.read_csv('/content/drive/My Drive/csv_files/gro_data_log.csv')

import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.pca import PCA

cols1 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder"]

x1 = data_log[cols1]
y = data_log["HELPFUL"]

logit_model = sm.Logit(y,x1)
result1 = logit_model.fit()
print(result1.summary())

cols2 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder", 
         "ReviewAuthorExperience", "ReviewAuthorReputation", "ReviewAuthorCredibility", "AverageRatingByReviewAuthor"]

x2 = data_log[cols2]
y = data_log["HELPFUL"]

logit_model = sm.Logit(y,x2)
result2 = logit_model.fit()
print(result2.summary())

cols3 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder", 
         "ReviewAuthorExperience", "ReviewAuthorReputation", "ReviewAuthorCredibility", "AverageRatingByReviewAuthor", 
         "ProductAverageRating", "ProductRank", "ProductPrice", "ProductTitleLength", "ProductDescriptionLength"]

x3 = data_log[cols3]
y = data_log["HELPFUL"]

logit_model = sm.Logit(y,x3)
result3 = logit_model.fit()
print(result3.summary())

cols4 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder", 
         "ReviewAuthorExperience", "ReviewAuthorReputation", "ReviewAuthorCredibility", "AverageRatingByReviewAuthor", 
         "ProductAverageRating", "ProductRank", "ProductPrice", "ProductTitleLength", "ProductDescriptionLength", 
         '"Beverages"', '"Confectionery"', '"Flavor"', '"ProductAppearance"']

x4 = data_log[cols4]
y = data_log["HELPFUL"]

logit_model = sm.Logit(y,x4)
result4 = logit_model.fit()
print(result4.summary())


#--------------------------------------------------#
# Plot #
#--------------------------------------------------#
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def coefplot(result):
    '''
    Takes in results of OLS model and returns a plot of 
    the coefficients with 95% confidence intervals.
    
    Removes intercept, so if uncentered will return error.
    '''
    # Create dataframe of results summary 
    coef_df = pd.DataFrame(result4.summary().tables[1].data)
    
    # Add column names
    coef_df.columns = coef_df.iloc[0]

    # Drop the extra row with column labels
    coef_df=coef_df.drop(0)

    # Set index to variable names 
    coef_df = coef_df.set_index(coef_df.columns[0])

    # Change datatype from object to float
    coef_df = coef_df.astype(float)

    # Get errors; (coef - lower bound of conf interval)
    errors = coef_df['coef'] - coef_df['[0.025']
    
    # Append errors column to dataframe
    coef_df['errors'] = errors

    # Sort values by coef ascending
    coef_df = coef_df.sort_values(by=['coef'])

    ### Plot Coefficients ###

    # x-labels
    variables = list(coef_df.index.values)
    
    # Add variables column to dataframe
    coef_df['variables'] = variables
    
    # Set sns plot style back to 'poster'
    # This will make bars wide on plot
    sns.set_context("poster")

    # Define figure, axes, and plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Error bars for 95% confidence interval
    # Can increase capsize to add whiskers
    c = np.array(["mediumaquamarine","palevioletred","palevioletred","orange",
                  "cornflowerblue","cornflowerblue","cornflowerblue",
                  "mediumaquamarine","mediumaquamarine","mediumaquamarine",
                  "cornflowerblue","mediumaquamarine","orange","orange",
                  "palevioletred","palevioletred","palevioletred","orange"])
    
    coef_df.plot(x='variables', y='coef', kind='bar',
                 ax=ax, color='none', fontsize=16, 
                 ecolor=c, capsize=0,
                 yerr='errors', legend=False)
    
    # Set title & labels
    plt.title('Coefficients of Features w/ 95% Confidence Intervals',fontsize=20)
    ax.set_ylabel('Coefficients',fontsize=18)
    ax.set_xlabel('',fontsize=18)
    
    # Coefficients
    ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
               marker='o', s=200, 
               y=coef_df['coef'], color=c)
    
    # Line to define zero on the y-axis
    ax.axhline(y=0, linestyle='--', color='black', linewidth=2)
    
    # Add legend
    legend_elements = [Line2D([0],[0],marker='o',color='palevioletred',
                         label='Review Characteristics',markersize=10),
                       Line2D([0],[0],marker='o',color='orange',
                         label='Review Author Characteristics',markersize=10),
                       Line2D([0],[0],marker='o',color='mediumaquamarine',
                         label='Product Listing Characteristics',markersize=10),
                       Line2D([0],[0],marker='o',color='cornflowerblue',
                         label='Key Product Aspects',markersize=10)]
    ax.legend(handles=legend_elements, prop={'size': 14})

    return plt.show()

coefplot(result4)