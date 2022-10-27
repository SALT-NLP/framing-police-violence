from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
import gensim.downloader as api
import pickle as pkl
import pandas as pd
import numpy as np
import neuralcoref
import multiprocessing
import json, re, os, spacy, nltk
from unidecode import unidecode
import argparse
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

SUBJECTS = ['nsubj', 'nsubjpass']
OBJECTS = ['dobj', 'iobj', 'obj', 'obl', 'advcl', 'pobj']
MODIFIERS = ['amod', 'nn', 'acl']

def coref_preprocess(txt, max_sent_length=1000, max_doc_length=10000):
    txt = unidecode(txt.strip())
    return ' '.join([sent for sent in nltk.tokenize.sent_tokenize(txt) if len(sent)<=max_sent_length])[:max_doc_length]

def download_word2vec_embeddings():
    print("Downloading pre-trained word embeddings from: word2vec-google-news-300.\n" 
          + "Note: This can take a few minutes.\n")
    wv = api.load("word2vec-google-news-300")
    print("\nLoading complete!\n" +
          "Vocabulary size: {}".format(len(wv.vocab)))
    return wv

def get_name_str_set(name):
    return set([n.lower() for n in name.split(" ")])

def get_race_gender_str_set(race_gender):
    try:
        rg = "_".join(race_gender.split(" ")).lower()
        return set([x.strip().lower() for x in open('resources/racial_and_gender_lexicons_basic/%s.txt' %rg, 'r').readlines()])
    except:
        return set()

def token_is_victim(token, name, race, gender):
    victim_set = get_name_str_set(name).union(get_race_gender_str_set(race)).union(get_race_gender_str_set(gender))
    return token.lower_ in victim_set

OFFICER_REGEX = re.compile(r'police|officer|\blaw\b|\benforcement\b|\bcop(?:s)?\b|sheriff|\bpatrol(?:s)?\b|\bforce(?:s)?\b|\btrooper(?:s)?\b|\bmarshal(?:s)?\b|\bcaptain(?:s)?\b|\blieutenant(?:s)?\b|\bsergeant(?:s)?\b|\bPD\b|\bgestapo\b|\bdeput(?:y|ies)\b|\bmount(?:s)?\b|\btraffic\b|\bconstabular(?:y|ies)\b|\bauthorit(?:y|ies)\b|\bpower(?:s)?\b|\buniform(?:s)?\b|\bunit(?:s)?\b|\bdepartment(?:s)?\b|agenc(?:y|ies)\b|\bbadge(?:s)?\b|\bchazzer(?:s)?\b|\bcobbler(?:s)?\b|\bfuzz\b|\bpig\b|\bk-9\b|\bnarc\b|\bSWAT\b|\bFBI\b|\bcoppa\b|\bfive-o\b|\b5-0\b|\b12\b|\btwelve\b')
def token_is_officer(span):
    return len(OFFICER_REGEX.findall(str(span).lower()))>0

human_nouns = set(pd.read_csv('resources/textbook_analysis/people_terms.csv', names=['noun', 'race/gender', 'category'])['noun'].values) 
def token_is_human(token):
    return ((token.lower_ in human_nouns) and (token.pos_ in ['NOUN','PRON','PROPN']))

def is_victim(cluster, name, race, gender, check_human=False):
    if check_human:
        if not is_human(cluster):
            return False
    for span in cluster:
        for token in span:
            if token_is_victim(token, name, race, gender):
                return True
    return False

def is_officer(cluster, check_human=False):
    if check_human:
        if not is_human(cluster):
            return False
    for span in cluster:
        if token_is_officer(span):
            return True
    return False

def is_human(cluster):
    for span in cluster:
        for token in span:
            if token.pos_ in ['PROPN', 'PRON']:
                return True
            if token.ent_type_ == "PERSON":
                return True
            if token_is_human(token):
                return True
    return False

def partition_tokens(doc, victimName, victimGender, victimRace, verbose):
    officer_tokens = set()
    victim_tokens = set()
    
    for token in doc:
        if token_is_officer(token):
            officer_tokens.add(token)
        if token_is_victim(token, victimName, victimRace, victimGender):
            victim_tokens.add(token)
    
    for cluster in doc._.coref_clusters:
        if is_officer(cluster):
            officer_tokens.update(set([token for span in cluster for token in span]))
        elif is_victim(cluster, victimName, 'ignore_gender', 'ignore_race', check_human=True) :
            victim_tokens.update(set([token for span in cluster for token in span]))
    
    if verbose:
        print('officer_tokens', officer_tokens)
        print('victim_tokens', victim_tokens)
    return officer_tokens, victim_tokens

    
def get_verbs_and_objects(subject, target_set=set()):

    def get_pobj(prep):
        for child in prep.children:
            if child.dep_ in OBJECTS:
                return child#.lower_
        return None

    def populate(verb, vo):
        for child in verb.children:
            if child.dep_ in OBJECTS:
                if child in target_set:
                    vo.append((verb, 'TARGET'))
                else:
                    vo.append((verb,child))
            elif child.dep_ == 'prep':
                pobj = get_pobj(child)
                if pobj: vo.append((verb, pobj))
            if child.dep_ in ['conj', 'xcomp']:
                populate(child, vo)
    vo = []
    populate(subject.head, vo)
    return vo

def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2))

class FrameExtractor(object):
    
    def __init__(self):
        legal_set = set([x.strip() for x in open('resources/legal_language/legal.txt', 'r').readlines()[7:]])
        self.legal_regex = r'(\b' + r'\b|\b'.join(list(legal_set)) + r'\b)'
        
        mental_set = set([x.strip() for x in open('resources/empath/mental_illness.txt', 'r').readlines()])
        self.mental_regex = r'(\b' + r'\b|\b'.join(list(mental_set)) + r'\b)'
        
        crime_set = set([x.strip() for x in open('resources/empath/crime.txt', 'r').readlines()[6:]])
        self.crime_regex = r'(\b' + r'\b|\b'.join(list(crime_set)) + r'\b)'

        self.word_embeddings = download_word2vec_embeddings()
        self.moral_concepts = self.get_moral_concepts()
        
    def extract_frames(self, df):
        extracted_frames = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            text = coref_preprocess(row['text'])
            name = row['name']
            gender = row['gender']
            age = row['age']
            race = row['race']
            weapons = set([x.lower() for x in literal_eval(row['weapons']) if len(x) > 0])
            try:
                doc = nlp(text)
            except Exception as e:
                print(e)
                extracted_frames[i] = {
                    'found.legal_language': None,
                    'found.mental_illness': None,
                    'found.criminal_record': None,
                    'found.fleeing': None,
                    'found.video': None,
                    'found.age': None,
                    'found.gender': None,
                    'found.unarmed': None,
                    'found.armed': None,
                    'found.race': None,
                    'found.official_report': None,
                    'found.interview': None,
                    'found.attack': None,
                    'found.systemic': None,
                    'found.victim_agentless_passive': None,
                    'found.victim_agentive_passive': None,
                    'found.victim_tokens': [],
                    'found.officer_tokens': [],
                    'victim_agentive_passive_heads': [],
                    'victim_agentive_officer_passive_heads': [],
                    'victim_agentless_passive_heads': [],
                    'num_words': 0,
                    'error': True
                }
                continue
                
            officer_tokens, victim_tokens = partition_tokens(doc, name, gender, race, verbose=False)
            interviews = self.interview(doc, officer_tokens, victim_tokens)
            agent_passive, agent_passive_heads, agent_officer_passive_heads, agentless_passive, agentless_passive_heads = self.victim_passive_frames(doc, victim_tokens, officer_tokens)
            extracted_frames[i] = {
                'found.legal_language': self.mentions_legal(text),
                'found.mental_illness': self.mentions_mental(text),
                'found.criminal_record': self.mentions_criminal(text),
                'found.fleeing': self.mentions_fleeing(text),
                'found.video': self.mentions_video(text),
                'found.age': self.mention_age(text, age),
                'found.gender': self.mention_gender(text, gender),
                'found.unarmed': self.is_unarmed(text),
                'found.armed': self.is_armed(doc, weapons),
                'found.race': self.mention_race(officer_tokens, victim_tokens, race),
                'found.official_report': interviews[0],
                'found.interview': interviews[1],
                'found.attack': self.mention_attack(doc, officer_tokens, victim_tokens, list(weapons)),
                'found.systemic': self.systemic(doc, officer_tokens, victim_tokens),
                'found.victim_agentless_passive': agentless_passive,
                'found.victim_agentive_passive': agent_passive,
                'found.victim_tokens': victim_tokens,
                'found.officer_tokens': officer_tokens,
                'victim_agentive_passive_heads': agent_passive_heads,
                'victim_agentive_officer_passive_heads': agent_officer_passive_heads,
                'victim_agentless_passive_heads': agentless_passive_heads,
                'num_words': len(doc),
                'error': False
            }
            self.extract_moral_frames(extracted_frames[i], doc, officer_tokens, victim_tokens)
            
        return pd.merge(df, pd.DataFrame().from_dict(extracted_frames, orient='index'), left_index=True, right_index=True)
    
    def mentions_legal(self, text):
        match = re.search(self.legal_regex, text.lower())
        if match:
            return match.span()[0]
        return -1
    
    def mentions_mental(self, text):
        match = re.search(self.mental_regex, text.lower())
        if match:
            return match.span()[0]
        return -1
    
    def mentions_criminal(self, text):
        match = re.search(self.crime_regex, text.lower())
        if match:
            return match.span()[0]
        return -1

    def mentions_fleeing(self, text):
        match = re.search(r'(\bflee(:?ing)?\b|\bfled\b|\bspe(?:e)?d(?:ing)? (?:off|away|toward|towards)|(took|take(:?n)?) off|desert|(?:get|getting|got|run|running|ran) away|pursu(?:it|ed))', text.lower())
        if match:
            return match.span()[0]
        return -1

    def mentions_video(self, text):
        match = re.search(r'(body(?: )?cam|dash(?: )?cam)', text.lower())
        if match:
            return match.span()[0]
        return -1
    
    def mention_age(self, text, age):
        match = re.search(r'\b%s\b'%age, text.lower())
        if match:
            return match.span()[0]
        return -1
    
    def mention_gender(self, text, gender):
        total_length = len(text)
        text = text[:int(total_length/3)]
        match = re.search(r'\b(woman|girl|daughter|mother|sister|female)\b', text.lower())
        if gender == 'Male':
            match = re.search(r'\b(man|boy|son|father|brother|male)\b', text.lower())
        if match:
            return match.span()[0]
        return -1
    
    def is_unarmed(self, text):
        match = re.search(r'unarm(?:ed|ing|s)?', text.lower())
        if match:
            return match.span()[0]
        return -1
    
    def is_armed(self, doc, weapons=set()):
        weapons = weapons.difference(set(['vehicle', '']))
        for token in doc:
            stripped = re.sub('[^\w]', '', token.lower_)
            if stripped in weapons:
                return token.idx
            if len(stripped)>0:
                if re.match(r'^arm(ed|ing|s)?$', stripped) and (token.pos_ !='NOUN'):
                    return token.idx
        return -1
    
    def mention_race(self, off, vic, race, verbose=False):
        race_set = get_race_gender_str_set(race)
        for token in vic:
            for child in token.head.children:
                if child.lower_ in race_set:
                    if verbose: print(token.lower_, child.lower_, child.dep_)
                    return child.idx
        return -1

    def interview(self, doc, off, vic, verbose=False):
        say = ['say', 'tell', 'explain', 'report', 'answer', 'claim', 'declare', 'reply', 'state', 
               'confirm']
        subjects = ['nsubj', 'nsubjpass']

        official_idx = -1
        commoner_idx = -1

        for token in doc:
            if ((token.head.head.lower_ == 'according') or \
            ((token.head.lemma_ in say) and (token.dep_ in subjects))):
                if (token in off) or (token.lemma_ in ['investigator', 'authority', 'source', 'official']):
                    if verbose: print('official', token)
                    official_idx = token.head.idx
                elif (token not in vic) and ((token.ent_type_=='PERSON') or (token.pos_ =='PRON') or (str(token) in ['man', 'woman', 'he', 'she']) ):
                    if verbose: print('commoner', token)
                    commoner_idx = token.head.idx
        return official_idx, commoner_idx

    
    def mention_attack(self, doc, off, vic, weapons, verbose=False):
        attack_verbs = ['shoot', 'fire', 'stab', 'lunge', 'confront', 'attack', 'strike', 'injure', 'harm']
        attack_objects = weapons + ['weapon', 'gun']

        for token in doc:
            if token.dep_ == 'nsubj':
                if token in vic:
                    for verb, obj in get_verbs_and_objects(token, target_set=off):
                        if type(obj) == str:
                            if (obj == 'TARGET') and verb.lemma_ in ['drive', 'accelerate', 'advance']:
                                if verbose: print(verb)
                                return verb.idx
                        else:
                            if verb.lemma_ in attack_verbs:
                                if verbose: print(verb)
                                return verb.idx
                            if (obj.lemma_ in attack_objects):
                                if verbose: print(verb)
                                return verb.idx
                else:
                    for verb, obj in get_verbs_and_objects(token, target_set=off):
                        if (verb.lemma_ in attack_verbs) and (str(obj)=='TARGET'):
                            if verbose: print(verb)
                            return verb.idx
        return -1
    
    def systemic(self, doc, off, vic, verbose=False):
        def victim_subject(obj):
            for child in obj.head.children:
                if (child.dep_=='nsubj') and (child in vic):
                    return True
            return False

        for token in doc:
            if token.dep_ in ['nsubjpass', 'dobj', 'iobj', 'obj']:
                if (token.head.lemma_ in ['shoot', 'kill', 'murder']) and (token not in vic) and (token not in off) and (token.ent_type_ == "PERSON"):
                    if not victim_subject(token):
                        if verbose: print(token.head, token.dep_, token)
                        return token.head.idx
        match = re.search(r'(nation(?:[ -])?wide|wide(?:[ -])?spread|police violence|police shootings|police killings|racism|racial|systemic|reform|no(?:[ -])?knock)', str(doc).lower())
        if match:
            if verbose: print(match)
            return match.span()[0]
        return -1
    
    
    def victim_passive_frames(self, doc, victim_tokens, officer_tokens):
        victim_agent_passive = np.inf
        victim_agentless_passive = np.inf

        victim_agent_passive_heads = []
        victim_agent_officer_passive_heads = []
        victim_agentless_passive_heads = []
        for token in doc:
            if token in victim_tokens: # victim is the patient / subject here
                if token.dep_ == 'nsubjpass': # passive

                    has_agent = np.any(np.array([child.dep_ == 'agent' for child in token.head.children]))

                    if has_agent:
                        victim_agent_passive = min(victim_agent_passive, token.head.idx)
                        victim_agent_passive_heads.append((token.head.lower_, token.head.idx))

                        officer_agent = np.any(np.array([(child.dep_ == 'agent') and (len(set(child.children).intersection(officer_tokens))>0) for child in token.head.children]))
                        if officer_agent:
                            victim_agent_officer_passive_heads.append((token.head.lower_, token.head.idx))
                    else:
                        victim_agentless_passive = min(victim_agentless_passive, token.head.idx)
                        victim_agentless_passive_heads.append((token.head.lower_, token.head.idx))

        return victim_agent_passive, victim_agent_passive_heads, victim_agent_officer_passive_heads, victim_agentless_passive, victim_agentless_passive_heads
    
    def extract_moral_frames(self, frame_dict, doc, off, vic):
        def get_all_verbs_and_modifiers(token):
            vm = []
            if token.dep_ == 'nsubj':
                vm.append(token.head)
            for child in token.children:
                if child.dep_ in MODIFIERS:
                    vm.append(child)
            return vm
        
        officer_vm = [] # verbs and modifiers
        for token in off: officer_vm.extend(get_all_verbs_and_modifiers(token))
            
        victim_vm = [] # verbs and modifiers
        for token in vic: victim_vm.extend(get_all_verbs_and_modifiers(token))
            
        for who, vm in zip(['officer', 'victim'], [officer_vm, victim_vm]):
            for concept in self.moral_concepts:
                matches = [x for x in vm if x.lower_ in self.moral_concepts[concept]]
                if len(matches):
                    try:
                        text=str(doc)
                        idx_1 = matches[0].idx
                        idx_2 = idx_1+len(matches[0])
                        print(who, concept, matches, text[max(0,idx_1-100):min(idx_2+100,len(text))])
                    except:
                        print(who, concept, matches)
                frame_dict['%s.%s' % (who, concept)] = len(matches)
                frame_dict['found.%s.%s' % (who, concept)] = matches
    
    def get_moral_concepts(self):
        morals = defaultdict(set)
        with open('resources/moral_foundations/Enhanced_Morality_Lexicon_V1.1.txt', 'r') as infile:
            for line in infile.readlines():
                split = line.split('|')
                token = split[0][8:]
                moral_foundation = split[4][9:]
                morals[moral_foundation].add(token)
                
        return dict(morals)
    
def worker(w_i):
    try:
        start_idx = int((w_i/NUM_JOBS)*TOTAL_LENGTH)
        end_idx = int(((w_i+1)/NUM_JOBS)*TOTAL_LENGTH)

        print('working between', start_idx, 'and', end_idx)

        data_slice = data.iloc[start_idx:end_idx]
        
        frames = F.extract_frames(data_slice)
        frames.to_csv('frames-extracted/frames_%s_%s.csv' % (start_idx, end_idx))
    except Exception as e:
        print('worker', w_i, 'quit with exception:')
        print(e)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input file', default='data/prepared/shootings/all_dated.csv')
    args = parser.parse_args()
    
    F = FrameExtractor()
    print("loading data...")
    data = pd.read_csv(args.input)
    
    NUM_JOBS = 1
    TOTAL_LENGTH = len(data)
    
    MULTI = False
    if MULTI:
        jobs = []
        for i in range(NUM_JOBS):
            if i not in [0,1]:
                continue
            p = multiprocessing.Process(target=worker, args=(i,))
            jobs.append(p)
            p.start()
    else:
        for w_i in [0]:
            start_idx = int((w_i/NUM_JOBS)*TOTAL_LENGTH)
            end_idx = int(((w_i+1)/NUM_JOBS)*TOTAL_LENGTH)

            print('working between', start_idx, 'and', end_idx)

            data_slice = data.iloc[start_idx:end_idx]

            frames = F.extract_frames(data_slice)
            frames.to_csv('frames-extracted/frames_%s_%s.csv' % (start_idx, end_idx))