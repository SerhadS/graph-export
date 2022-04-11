import networkx as nx
from nltk.corpus import wordnet as wn
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import re
import requests
import json
import numpy as np
import copy
import csv
from collections import Counter
import string
import time
from nltk.corpus import stopwords




#Initialize punctuation remover and lemmatizer
punct = string.punctuation
temp = []
for i in range(len(punct)):
    if punct[i]=='-' or punct[i]=='/':
        temp.append(i)
temp = sorted(temp, reverse=True)
for i in temp:
    punct = punct[:i]+punct[i+1:]
translator = str.maketrans(punct, ' '*len(punct))

def custom_tokenizer(nlp):
    '''
        Custom tokenizer for not to split words which include "-" and "/"
    '''
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = custom_tokenizer(nlp)




class GraphExportAgent():
    """_summary_
    """

    def __init__(self, text_path, cfg = None, model = None):
        """
        :param text_path: the txt file path to be read
        :type text_path: string
        :param cfg: class configuration (defaults to None)
        :type cfg: dict
        :param model: a gensim model to be required to calculate semantic similarities between words
        :type model: gensim.models.KeyedVectors
       
        """
    

        default_cfg = {
            'USPTO_stopwords_path': './USPTO_stopwords.csv',
            'technet_stopwords_path': './technical_stopwords.csv',
            'technet_add_stopwords_path': './tn_additional_stopwords.csv',
            'selected_kb': 'wordnet' # in ['technet', 'wordnet', 'conceptnet', 'other']
        }
        if cfg:
            assert all(key in default_cfg.keys() for key in cfg.keys())
            default_cfg = cfg
        self.cfg = default_cfg

        if self.cfg['selected_kb'] not in ['technet', 'wordnet', 'conceptnet']:
            print(f'WARNING: The selected knowledge base should be on of {["technet", "wordnet", "conceptnet"]}, defaulting to technet')
            self.cfg['selected_kb']= 'technet'


        with open(self.cfg['USPTO_stopwords_path'], encoding = 'utf-8') as f:
            reader = csv.reader(f)
            uspto_stops = [x[0].replace(' ', '_') for x in reader]
        with open(self.cfg['technet_stopwords_path'], encoding = 'utf-8') as f:
            reader = csv.reader(f)
            technical_stops = [x[0].replace(' ', '_') for x in reader]
        with open(self.cfg['technet_add_stopwords_path'], encoding = 'utf-8') as f:
            reader = csv.reader(f)
            tn_additional_stops = [x[0].replace(' ', '_') for x in reader]
        self.stops = set(stopwords.words('english') + uspto_stops + technical_stops + tn_additional_stops)

        if self.cfg in ['technet', 'concept']:
            if model:
                self.model = model
            else:
                warning = """
                TechNet and ConceptNet working modes require a word2vec model to be able to work\n
                Defaulting to WordNet working mode
                """
                print(warning)
                self.cfg['selected_kb'] = 'wordnet'
        
        self.text_path = text_path

        self.model = model


    
    def set_vocab(self):
        """sets the vocab attribute of the agent
        """
        if self.model:
            self.vocab = self.model.key_to_index
        else:
            self.vocab = {word:i for i,word in enumerate(wn.all_lemma_names())}

    
    def paragraphReader(self, path):
        """read the text from a txt file

        :param path: file path
        :type path: string
        :return: input string processed by spacy
        :rtype: spacy doc object
        """


        with open(path, 'r', encoding = 'utf-8') as f:
            sr = f.readlines()

        sr = " ".join(sr).lower().replace('\n', '')

        return nlp(sr)

    def get_chunks(self, doc):
        """get the set of chunks in doc object

        :param doc: spacy doc object of the text read
        :type doc: spacy doc object
        :return: list of unique chunks in text
        :rtype: list
        """
        
        chunks = []
        for chunk in doc.noun_chunks:
            if chunk[0].text in self.stops:
                chunk = chunk[1:]
            
                if chunk not in chunks:
                    chunks.append(chunk)

        return chunks


    def ifPhraseInSemanticNetwork (self, chunks):
        """finds if the chunks in chunks list are represented
        as a phrase in a semantic network (not wordnet)

        :param chunks: list that contains chunks
        :type chunks: list
        :return: the list of spacy chunks contained in model
        :rtype: list
        """
        chunks_CN = []
        for chunk in chunks:
            flag = 0
            for i in range(len(chunk)-1):
                if self.vocab.get(chunk[i:len(chunk)-1].text.replace(' ', '_')+'_'+chunk[-1].lemma_):
                    chunks_CN.append(chunk[i:len(chunk)-1].text.replace(' ', '_')+'_'+chunk[-1].lemma_)
                    flag = 1
                if flag == 1:
                    break
        return chunks_CN
    
    def ifPhraseInWordNet (self, chunks):
        """finds if the chunks in chunks list are represented
        as a phrase in wordnet

        :param chunks: list that contains chunks
        :type chunks: list
        :return: the list of spacy chunks contained in wordnet
        :rtype: list
        """
        chunks_WN = []
        for chunk in chunks:
            flag = 0
            for i in range(len(chunk)-1):
                if wn.synsets(chunk[i:len(chunk)-1].text.replace(' ', '_')+'_'+chunk[-1].lemma_):
                    chunks_WN.append(chunk[i:len(chunk)-1].text.replace(' ', '_')+'_'+chunk[-1].lemma_)
                    flag = 1
                if flag == 1:
                    break
        return chunks_WN
    

    def techNetPreprocessAPI(self, text):
        """automatically get the words and phrases contained in a text piece

        :param text: text to be analysed by TechNet API
        :type text: string
        :return: the API answer in json format
        :rtype: dict
        """


        API_URL_HOST = "http://52.221.86.148/api/ideation/concepts/preprocess1"
        userid = 'serhad'
        data = {'paragraph':text, 'userid':userid}
        r = requests.post(url = API_URL_HOST, json = data)

        return list(set([item for sublist in r.json()['processed'] for item in sublist]))


    def dependencyHelper(self, terms):
        """calculates a adjacency matrix where i,j is semantic similarity between terms[i] and terms[j]

        :param terms: a list of terms
        :type terms: list
        :return: adjacency matrix
        :rtype: np.array
        """
        mtx = np.zeros((len(terms), len(terms)))

        for i in range(len(mtx)):
            for j in range(i+1, len(mtx)):
                temp = self.term2termSimilarity(terms[i], terms[j])
                mtx[i][j] = temp if temp else 0
                mtx[j][i] = mtx[i][j]

        return mtx

    def term2termSimilarity(self, term1, term2):
        """calculates the semantic similarity of two terms

        :param term1: first term
        :type term1: str
        :param term2: second term
        :type term2: str
        :return: semantic similarity
        :rtype: float
        """

        if self.cfg['selected_kb'] == 'wordnet':
            return wn.path_similarity(wn.synsets(term1)[0], wn.synsets(term2)[0])
        else:
            return self.model.similarity(term1, term2)
    

    def main_logic(self):
        
        
        self.set_vocab()
        

        if self.cfg['selected_kb'] == 'technet':
            terms = self.techNetPreprocessAPI(doc.text)

        else:
            doc = self.paragraphReader(self.text_path)
            chunks = self.get_chunks(doc)

            if self.cfg['selected_kb'] == 'wordnet':
                chunks = self.ifPhraseInWordNet (chunks)

            elif self.cfg['selected_kb'] == 'conceptnet':
                chunks = self.ifPhraseInSemanticNetwork(chunks)

            chunks = set(sorted(chunks, key = lambda x:len(x.split('_')), reverse = True))

            text = doc.text
            for chunk in chunks:
                text = re.sub('\s'+chunk.replace('_', ' ')+"[a-z]*[\s,.]", ' '+chunk+' ', text)

            if self.cfg['selected_kb'] == 'wordnet':
                terms = [x for x in set([item.lemma_ for item in nlp(text) if wn.synsets(item.lemma_)]) if x not in self.stops and x.isdigit()==False and len(x)>1]

            elif self.cfg['selected_kb'] == 'conceptnet':
                terms = [x for x in set([item.lemma_ for item in nlp(text) if self.vocab.get(item.lemma_)]) if x not in self.stops and x.isdigit()==False and len(x)>1]


        adj = self.dependencyHelper(terms)

        graph, graph_mst, graph_mst_nextStrong = self.graphBuilder(adj, terms)

        return graph, graph_mst, graph_mst_nextStrong




    def graphBuilder(self, mtx, terms):
        """generates a graph given and adjacency mtx and node names

        :param mtx: adjacency matrix
        :type mtx: numpy.array
        :param terms: list of node names
        :type terms: list
        :return: graph : whole graph, graph_mst: MST of the graph, graph_mst_nextStrong: MST+(N-1) next strongest edges where N is number of nodes
        :rtype: networkx.Graph
        """
        
        # build the grapj
        graph = nx.from_numpy_matrix(mtx)
        nx.set_node_attributes(graph, {i:terms[i] for i in range(len(terms))}, 'name')
        
        # get the MST of the graph
        graph_mst = nx.algorithms.tree.mst.maximum_spanning_tree(graph, weight = 'weight')

        edges = [x for x in graph.edges.data()]
        edges = sorted(edges, key = lambda x:x[2]['weight'], reverse = True)

        # add next N-1 edges
        n_add = 2*len(graph_mst.nodes())-len(graph_mst.edges())
        graph_mst_nextStrong  = graph_mst.copy()
        count = 0
        for edge in edges:
            if graph_mst_nextStrong.has_edge(edge[0], edge[1]) == False:
                graph_mst_nextStrong.add_edge(edge[0], edge[1], weight = edge[2]['weight'])
                count+=1
            if count == n_add:
                break
        
        return graph, graph_mst, graph_mst_nextStrong




