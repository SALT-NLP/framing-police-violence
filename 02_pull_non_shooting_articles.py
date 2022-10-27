from datetime import date
from googlesearch import search 
from bs4 import BeautifulSoup
from bs4.element import Comment
import pandas as pd
import urllib.request
import requests
import json, datetime, re, time, os, glob

def get_query(row):
    day = datetime.datetime.strptime(row['date'], '%Y-%m-%d')
    day_plus_1 = day+datetime.timedelta(days=1)
    
    site = row['url'].replace("http://", "").replace("https://", "").split('/')[0]
    
    query = "news "
    query += " ".join(['-"%s"'% word for word in row["name"].split()])
    query += " site:%s after:%s before:%s" % (site,day.strftime('%Y-%m-%d'),day_plus_1.strftime('%Y-%m-%d'))
    return query

def save_url(url, out_fn):
    directory = os.path.dirname(out_fn)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        web = requests.get(url,timeout=10)
    except:
        print('\trequest error')
        return
    if web.status_code != 200:
        print('\t'+str(web.status_code), 'error')
        return
    html = web.content
    soup = BeautifulSoup(html, 'html.parser')
    
    with open(out_fn, 'w', encoding='utf-8') as outfile:
        outfile.write('\tURL:\t' + url + '\n')
        
        try:
            p = re.compile(r'^(\d+) bytes$')
            el = soup.find(text=p)
            size = p.match(el.string).group(1)
            if size > 1000000:
                outfile.write('too large')
                return
        except:
            pass
        
        outfile.write(str(soup))
        
def main():
    shootings = pd.read_csv('data/prepared/shootings/shooting_frames.csv')
    polarized = shootings[(shootings['leaning']==0) | (shootings['leaning']==2)].copy()
    polarized['TAG'] = ['%s_%s' % (row['id'], row['page_num']) for _, row in polarized.iterrows()]
    for _, row in polarized.iterrows():
        
        out_fn = "data/raw/no-shootings-control/%s.html" % row['TAG']
        if os.path.exists(out_fn):
            print('already saved', out_fn)
            continue
        
        print(row['url'])
        print(row['name'])
        print(get_query(row))
        print()
        try:
            for result in search(get_query(row), num_results=1): 
                try:
                    save_url(result, out_fn)
                except Exception as e:
                    print("error", e)
                time.sleep(1)
        except Exception as e:
            print("stopped for error [%s]. starting again in 20 minutes" % e)
            time.sleep(1200)
            
main()