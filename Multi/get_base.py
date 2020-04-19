import os, requests, urllib3, pickle
import string, re, nltk
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# nltk.download('stopwords'), nltk.download('punkt')
from rnnmorph.predictor import RNNMorphPredictor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
s = requests.Session()

if not os.path.isdir('data'):
    os.mkdir('data')
# Extract relevant page urls from each thread by topic

pairs = []


def get_urls_list(url, category_name):
    global pairs
    resp = s.get(url, verify=False)
    soup = BeautifulSoup(resp.text, 'html.parser')
    pagenav = soup.find('div', class_='pagenav')
    last_page_num = int(pagenav.find(nowrap='nowrap').find('a').get('href').split('=')[-1])
    urls_list = [(category_name, f'{url}&page={i}') for i in range(1, last_page_num+1)]
    pairs += urls_list


with open('data/threads.txt', 'r') as t:
    threads = [a.split() for a in t.read().split('\n') if a]
t.close()

for thread in threads:
    get_urls_list(thread[0], thread[1])

with open('data/links.pkl', 'wb') as links:
    pickle.dump(pairs, links)

# Extract texts from URLs, preprocess and save them

russian_stopwords = nltk.corpus.stopwords.words("russian")
predictor = RNNMorphPredictor(language="ru")


def preprocess_sentence(sen):
    sen = sen.rstrip().lower().replace('-то', ' -то').replace('-нибудь', ' -нибудь').replace('ё', 'е')
    for c in list(string.punctuation + '1234567890'):
        sen = sen.replace(c, ' ')
    sen = sen.replace('»', ' ').replace('«', ' ').replace('—', '').replace('–', '')
    sen = re.sub(u'[A-z]', u'', sen)
    sen = [i for i in sen.split() if i]

    tokens = [token for token in sen if token not in russian_stopwords]
    text = ' '.join([a.normal_form for a in predictor.predict(tokens)])
    return text


lines_list = []


def download_threads(pages, start=0):
    global lines_list

    for post_category, url in tqdm(pages[start:]):
        resp = s.get(url, verify=False)
        soup = BeautifulSoup(resp.text, 'html.parser')
        posts = soup.find('div', {'id': 'posts'}).findAll('div', class_='page')

        for post in posts:
            try:
                post_username = post.find('a', class_='bigusername').text
                post_text = preprocess_sentence(' '.join(post.find('td', class_='alt1').text.strip().split()))
                post_class = 'Anime'
                lines_list.append({'username': post_username, 'text': post_text, 'class': post_class})
            except:
                pass


if os.path.isfile('data/base.csv'):
    base = pd.read_csv('data/base.csv')
    checkpoint = len(base)
else:
    base = pd.DataFrame()
    checkpoint = 0

download_threads(pairs, checkpoint)
base = base.append(lines_list).reset_index(drop=True).dropna()


def quote_msg(x):
    x = x.replace('цитата сообщение ', '')
    return x


base['text'] = base['text'].apply(quote_msg)

base.to_csv('data/base.csv', index=False)
