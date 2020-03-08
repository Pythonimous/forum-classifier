#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm


# In[2]:


from rnnmorph.predictor import RNNMorphPredictor
predictor = RNNMorphPredictor(language="ru")


# In[3]:


def lemmatize(name):
    
    with open(name+'_corpus_clean.txt', 'r', encoding='utf-8') as a:
        s = a.read().split('\n')
    a.close()
        
    sents = [t.split() for t in s]
    
    lemmas = [' '.join([a.normal_form for a in predictor.predict(sen)]) for sen in tqdm(sents)]
    
    with open(name+'_lemmas.txt', 'w', encoding='utf-8') as b:
        b.write('\n'.join(lemmas))
    b.close()
    
    return lemmas


# In[4]:


lemmatize('Val')


# In[5]:


lemmatize('Test')


# In[6]:


lemmatize('Train')


# In[ ]:




