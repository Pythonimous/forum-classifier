import os, requests, urllib3, pickle
import string, re, nltk
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils import split_texts_to_len

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


def download_threads(base, pages):
    count = 0
    lines_list = []
    for post_category, url in tqdm(pages):
        resp = s.get(url, verify=False)
        soup = BeautifulSoup(resp.text, 'html.parser')
        posts = soup.find('div', {'id': 'posts'}).findAll('div', class_='page')

        for post in posts:
            try:
                post_username = post.find('a', class_='bigusername').text
                line = post.find('td', class_='alt1').text.strip()
                post_text = preprocess_sentence(' '.join(line.split()))
                post_class = post_category
                lines_list.append({'username': post_username,
                                   'text': post_text,
                                   'class': post_class,
                                   'page_url': url})
            except:
                pass
        count += 1
        if count == 50:
            base = add_to_base(base, lines_list)
            count = 0
            lines_list = []

            base.to_csv('data/complete_base.csv', index=False)

    base = add_to_base(base, lines_list)
    return base


def add_to_base(base, list_to_add):
    def quote_msg(x):
        x = x.replace('цитата сообщение ', '')
        return x
    base = base.append(list_to_add).reset_index(drop=True).dropna()
    base['text'] = base['text'].apply(quote_msg)
    return base


if os.path.isfile('data/complete_base.csv'):
    base = pd.read_csv('data/complete_base.csv')
    pairs = [p for p in pairs if p[1] not in set(base['page_url'])]
else:
    base = pd.DataFrame()

base = download_threads(base, pairs)
base = base.dropna()
base.to_csv('data/complete_base.csv', index=False)

row_lengths = []
for _, row in base.iterrows():
    row_lengths.append(len(row['text'].split()))
average_words = int(sum(row_lengths) / len(row_lengths))

resampled_base = split_texts_to_len(base, average_words, 42)
resampled_base.to_csv('data/resampled_base.csv', index=False)

truncated_base = resampled_base[resampled_base['text'].str.len() > 150]
truncated_base.to_csv('data/base.csv', index=False)
