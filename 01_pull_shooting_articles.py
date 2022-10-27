from datetime import date
from googlesearch import search 
from bs4 import BeautifulSoup
from bs4.element import Comment
import pandas as pd
import urllib.request
import requests
import json, datetime, re, time, os, glob

def get_before_after(datestr, days=30):
    aday, amonth, ayear = datestr.split('/')
    
    after = date(int(ayear), int(amonth), int(aday))
    after = after - datetime.timedelta(days=1)
    before = after + datetime.timedelta(days=days)

    return ("after:%s before:%s" % (after.strftime('%Y-%m-%d'),before.strftime('%Y-%m-%d')))

def get_query(row, regex=re.compile('[^a-zA-Z ]')):
    victims_name = regex.sub('', row["Victim's name"])
    query = '("%s" OR "%s %s")' % (victims_name, victims_name.split()[0], victims_name.split()[-1])
    query += " AND (shooting OR shot OR killed OR died OR fight OR gun)"
    query += " AND (police OR officer OR officers OR law OR enforcement OR cop OR cops OR sheriff OR patrol)"
    query += " " + get_before_after(row["Date of Incident (month/day/year)"])
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

df = pd.read_csv('data/raw/shootings/police_killings_MPV.csv')
for _, row in df.iterrows():
    if row["Victim's name"] not in ['Name withheld by police', 'This is a spacer for Fatal Encounters use.']:
        fn = 'data/raw/shootings-json/%s.json' % int(row['MPV ID'])
        if os.path.exists(fn):
            print('\talready saved')
            continue
        
        data = {}
        data['FEID'] = ['Fatal Encounters ID'] 
        data['name'] = row["Victim's name"]
        data['age'] = row["Victim's age"]
        data['gender'] = row["Victim's gender"]
        data['race'] = row["Victim's race"]
        data['primary-url'] = row["Link to news article or photo of official document"]
        data['query'] = get_query(row)
        data['urls'] = {}
        index = 1
        try:
            for result in search(data['query'], num_results=30): 
                data['urls'][index] = result
                try:
                    save_url(result, "data/raw/shootings-articles/%s/%s.html" % (int(row['MPV ID']), index))
                except Exception as e:
                    print("error", e)
                time.sleep(1)
                index += 1
        except:
            print("stopped for error. starting again in 20 minutes")
            time.sleep(1200)

        with open(fn, 'w') as outfile:
            json.dump(data, outfile)
            print('dumped', fn)
