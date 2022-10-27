import pandas as pd
from glob import glob
import json, tldextract
from tqdm import tqdm
from custom_date_extractor import *
import tldextract


data_dict = {}
fn_list = tqdm(sorted(glob('data/raw/shootings-txt/*/*.txt')))
for i, fn in enumerate(fn_list):
    page_num = fn.split('/')[-1][:-4]
    id_num = fn.split('/')[-2]
    
    with open(fn, 'r') as infile:
        text = infile.read()
        
    try:
        
        with open('data/raw/shootings-json/%s.json' % id_num, 'r') as infile:
            JSON = json.load(infile)
                
        with open('data/raw/shootings-articles/%s/%s.html' % (id_num, page_num), 'r') as infile:
            html = infile.read()
            lines = html.split('\n')
            url = lines[0].split('\t')[-1].strip()
            
        subdomain, domain, suffix = tldextract.extract(url)
        date = extractArticlePublishedDate(url, html)
        
    except Exception as e:
        print(e, fn)
        continue
    
    data_dict[i] = {
        'id': int(id_num),
        'page_num': page_num,
        'text':text,
        #'xurl': JSON['urls'][page_num],
        'url': url,
        'domain': domain,
        'subdomain': subdomain,
        'suffix': suffix,
        "name": JSON['name'], 
        "age": JSON['age'], 
        "gender": JSON['gender'],
        "race": JSON['race'],
        "article_date": date
    }
data = pd.DataFrame().from_dict(data_dict, orient='index')
data = data[[len(t)>0 for t in data.text.values]].copy()

mpv = pd.read_csv('data/prepared/shootings/MPV_clean.csv')
mpv['id'] = mpv['MPV ID']
data = pd.merge(data, mpv, on='id', suffixes=('', '_'))[['id', 'page_num', 'text', 'url', 'domain', 'subdomain', 'suffix',
                                                         'name', 'age', 'gender', 'race', 'date', 'article_date', 'address', 'city', 'state', 
                                                         'zip', 'county', 'agency', 'cause_of_death', 'description', 'outcome', 
                                                         'mental_illness', 'armed', 'weapons', 'attack', 'fleeing', 'video', 'off_duty', 'geography', 'MPV ID']]


bias = []
urls = []
media_bias = pd.read_csv('resources/news_sources/media-bias-fc-scrape.csv')
for i, row in media_bias.iterrows():
    val = row['bias_png']
    if 'center' in val:
        bias.append(1)
    elif 'left' in val:
        bias.append(0)
    elif 'right' in val:
        bias.append(2)
    else:
        bias.append(1)
    subdomain, domain, suffix = tldextract.extract(row['url'])
    urls.append(domain)
media_bias['bias'] = bias
media_bias['domain'] = urls

from tqdm import tqdm
leanings = []
for i, row in tqdm(data.iterrows(), total=len(data)):
    bias = media_bias[media_bias['domain']==row['domain']]
    if len(bias)>0:
        leanings.append(bias.iloc[0]['bias'].item())
    else:
        leanings.append(-1)

data['leaning'] = leanings

df.to_csv('data/prepared/shootings/all_dated.csv')