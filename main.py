import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize,word_tokenize

sentance = "Hello i am Dr. ashish prajapati. I am a student"
sent_tokenized = sent_tokenize(sentance)

print("first sentance", sent_tokenized[0])

word_token = word_tokenize(sentance)
print(word_token)

# steaming
from nltk import PorterStemmer
steamer = PorterStemmer()

import time
start_time = time.time()

words = [
    'run', 'ran', 'running', 'runner', 'runs',
    'eat', 'eating', 'eaten', 'eats',
    'play', 'played', 'playing', 'player',
    'write', 'writing', 'written', 'writes',
    'study', 'studied', 'studying', 'studies',
    'connect', 'connected', 'connecting', 'connection',
    'argue', 'argued', 'arguing', 'argument',
    'compute', 'computer', 'computing', 'computation',
    'organize', 'organized', 'organizing', 'organization',
    'analyze', 'analyzed', 'analyzing', 'analysis',
    'go', 'gone', 'going', 'goes',
    'swim', 'swimming', 'swam', 'swimmer',
    'drive', 'driving', 'drove', 'driver',
    'speak', 'speaking', 'spoken', 'speaker',
    'create', 'created', 'creating', 'creation',
    'develop', 'developed', 'developing', 'development',
    'manage', 'managed', 'managing', 'management',
    'react', 'reacted', 'reacting', 'reaction',
    'code', 'coded', 'coding', 'coder',
    'debug', 'debugged', 'debugging',
    'test', 'tested', 'testing', 'tester'
]

steamed = [steamer.stem(w) for w in words]

end_time = time.time()
time_elapsed = (end_time - start_time)


print(f'og words {words}')
print(f'steamed words {steamed}')
print(f'time elapsed {time_elapsed:3f}')


print(set(steamed))
print(len(set(steamed)))

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

lemetizer = WordNetLemmatizer()


start_time_lemetizer = time.time()
# run lemetization
lemetized = [lemetizer.lemmatize(w) for w in words]

end_time_lemetizer = time.time()



print(f'og words {words}')
print(f'lemetized words {lemetized}')
print(f'time elapsed {time_elapsed:3f}')

unique_lemetiz = set(lemetized)
print(unique_lemetiz)
print(len(unique_lemetiz))

from nltk.corpus import stopwords
nltk.download('stopwords')

stopword_list = set(stopwords.words('english'))
print(stopword_list)


text =  'Natural language processing is one of the most interesting fields in computer science. It allows computers to understand, analyze, and generate human language in a meaningful way. Many applications of natural language processing are used in chatbots, search engines, recommendation systems, and machine translation.  The main goal of NLP is to make communication between humans and machines easier and more effective. In recent years, the development of artificial intelligence and deep learning has improved the accuracy of language models significantly. However, the process of cleaning and preparing text data is still an important step before training any machine learning model. A data scientist often removes stopwords from text because these words do not add much meaning to the analysis. Words such as "the", "is", "at", "which", "on", and "for" appear very frequently in documents. Removing them can help improve performance and reduce unnecessary data during processing.'


print(text)

text_wt = word_tokenize(text)
print(text_wt)

text_wt_without_stopwords = [ w for w in text_wt if w.lower() not in stopword_list]
print(text_wt_without_stopwords)

import spacy

from spacy import displacy
# load spacy eng mode
nlp = spacy.load("en_core_web_sm")

text2 = "london is the capital and nost populated city."
print(text2)

doc = nlp(text2)
doc

displacy.render(doc,style='dep',jupyter=True,options={'distance':100})

for token in doc:
  print(f'{token.text:10} | POS: {token.pos_:10} | Tag: {token.tag_} | explanation: {spacy.explain(token.tag_)}')