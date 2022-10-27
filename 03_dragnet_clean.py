from dragnet import extract_content
from bs4 import BeautifulSoup
from glob import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_glob', type=str, default='data/raw/shootings-articles/*/*.html')
parser.add_argument('--output', type=str, default='data/raw/shootings-txt')
args = parser.parse_args()

fn_list = sorted(glob(args.input_glob))
for i, html_fn in enumerate(fn_list):
    page_num = html_fn.split('/')[-1][:-5]
    id_num = html_fn.split('/')[-2]
    
    print('cleaning', id_num, page_num)
    
    
    content_dir = '%s/%s' % (args.output, id_num)
    if not os.path.exists(content_dir):
        os.makedirs(content_dir)
    content_fn = "%s/%s.txt" % (content_dir, page_num)
        
    try:
        with open(html_fn, 'r') as infile:
            html = infile.read()
            soup = BeautifulSoup(html, 'html.parser')
            title = ' '.join([h.get_text() for h in soup.find_all('h1')])
            content = extract_content(html)
            with open(content_fn, 'w') as outfile:
                outfile.write(title)
                outfile.write('\n')
                outfile.write(content)
    except Exception as e:
        print(e)
    
    if i%100==0:
        print(i, '/', len(fn_list))