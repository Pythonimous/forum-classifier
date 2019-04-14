import time
from os import listdir
import re
from gensim.models import doc2vec, utils
from pymystem3.mystem import Mystem
mystem = Mystem()

def formattext(text):
    lemmas = mystem.lemmatize(text)
    text = ''.join(lemmas)
    return re.findall('\w+', text.lower())

docLabels = [f for f in listdir('news_life') if f.endswith('.txt')]

print(time.ctime())
print('preprocessing data')
data = []
for i in range(len(docLabels)):
    if not i % 10:
        print(i + 1, '/', len(docLabels))
    data.append(doc2vec.TaggedDocument(
            formattext(open('news_life/'+docLabels[i], encoding='utf-8').read()),
                       [docLabels[i]]))

print(time.ctime())
print('training model')
model = doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(data)
model.train(data, total_examples=model.corpus_count, epochs=model.iter)
print(time.ctime())
print('done training model')
model.save('life2vec')
#model = doc2vec.Doc2Vec.load('life2vec')
inferred_vector = model.infer_vector(data[0].words)
test_vector = model.infer_vector(formattext(open('1084713.txt', encoding='utf-8').read()))
print(test_vector)
print(data[0].tags)
print('Most similar:')
print(model.docvecs.most_similar([inferred_vector], topn=10))
print(model.docvecs.most_similar([test_vector], topn=10))
