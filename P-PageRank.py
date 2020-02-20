import numpy as np
from scipy import sparse
"""
A few functions that needs to be done:
    - preprocessing
    - build citation graph
    - compute pagerank
"""

def preprocessing():
    """
    dictionary 1: paper index to paper index it cites
    paper index to conferences
    parse the raw dataset, 
    extract useful information, 
    return the parsed entities.
    """
    class node:
        def __init__(self, selfind=None):
            self.selfind = selfind
            self.conf = None
            self.citations = []
    conferences, papers = {},{}
    confset = set()
    f = open("DBLPOnlyCitationOct19.txt","r")
    curnode = node()
    for line in f:
        if line[0:7] == "1632442":
            curnode = node()
        elif line == "\n":
            papers[curnode.selfind] = curnode.citations
            conferences[curnode.selfind] = curnode.conf
            curnode = node()
        elif line[0:2] == "#c":
            curnode.conf = line[2:].replace('\n','')
            confset.add(line[2:].replace('\n',''))
        elif line[0:6] == "#index":
            curnode.selfind = int(line[6:].replace('\n',''))
        elif line[0:2] == "#%":
            curnode.citations.append(int(line[2:].replace('\n','')))
    papers[curnode.selfind] = curnode.citations
    conferences[curnode.selfind] = curnode.conf   
    return conferences, papers, confset # or other objects

def build_conference_citation_net(conferences,papers):
    """
    build conference citation network
    
    return list of (conf1, conf2, weight) triples
    """
    
    #{0: 'topic one', 2232: 'hello'} {0: [], 2232: [0]}
    network = {}
    triples = []
    for key, value in papers.items():
        keyconf = conferences[key]
        if keyconf not in network:
            network[keyconf] = {}
        for ind in value:
            indconf = conferences[ind]
            if indconf in network[keyconf]:
                network[keyconf][indconf] += 1
            else:
                network[keyconf][indconf] = 1
    for key1, value1 in network.items():
        for key2,value2 in network[key1].items():
            tup = (key1,key2,value2)
            triples.append(tup)
    
    return triples

def pagerank(network,confset):
    """
    compute pagerank score
    """
    n = len(confset)
    M = np.zeros((n,n))
    conftoind = {}
    indtoconf = {}
    i = 0
    for con in confset:
        conftoind[con] = i 
        indtoconf[i] = con
        i = i + 1
        
    KDD_ind = conftoind["KDD"]
    ICML_ind = conftoind["ICML"]
    for tup in network:
        (conf1,conf2,weight) = tup
        conf1_ind = conftoind[conf1]
        conf2_ind = conftoind[conf2]
        M[conf1_ind][conf2_ind] = weight
    D = np.zeros((n,n))
    for i in range(0,n):
        sum = 0
        for num in M[i]:
            sum += num
        D[i][i] = sum
    for i in range(0,n):
        for j in range(0,n):
            if D[i][i] == 0:
                M[i][j] = 0
            else:
                M[i][j] = 1.0/D[i][i]*M[i][j]
            
    r_kdd = np.zeros((n,1))
    for i in range(0,n):
        if i == KDD_ind:
            r_kdd[i] = 1
            
    r_icml = np.zeros((n,1))
    for i in range(0,n):
        if i == ICML_ind:
            r_kdd[i] = 1
            
    r_both = np.zeros((n,1))
    for i in range(0,n):
        if i == ICML_ind:
            r_both[i] = 1
        elif i == KDD_ind:
            r_both[i] = 1
        
    
    M = M.transpose()
    
    M = sparse.csr_matrix(M)
    
    constvec = np.zeros((n,1))
    for i in range(0,n):
        if i == KDD_ind:
            constvec[i] = 0.2
    for _ in range(0,300):
        calc = sparse.csr_matrix.dot(M,r_kdd)
        calc = 0.8*calc
        for i in range(0,n):
            r_kdd[i] = constvec[i] + calc[i]
    
    constvec = np.zeros((n,1))
    for i in range(0,n):
        if i == ICML_ind:
            constvec[i] = 0.2
    for _ in range(0,300):
        calc = sparse.csr_matrix.dot(M,r_icml)
        calc = 0.8*calc
        for i in range(0,n):
            r_icml[i] = constvec[i] + calc[i]
    
    constvec = np.zeros((n,1))
    for i in range(0,n):
        if i == ICML_ind:
            constvec[i] = 0.1
        elif i == KDD_ind:
            constvec[i] = 0.1
    for _ in range(0,300):
        calc = sparse.csr_matrix.dot(M,r_both)
        calc = 0.8*calc
        for i in range(0,n):
            r_both[i] = constvec[i] + calc[i]
    
   
    return r_kdd, r_icml,r_both, indtoconf


def main():
    conferences,papers,confset = preprocessing()
    network = build_conference_citation_net(conferences,papers)
    r_kdd, r_icml,r_both,indtoconf = pagerank(network,confset)
    n = len(confset)
    print "for top-10 most similar conferences to KDD"
    ranking = {}
    for i in range(0,n):
        if indtoconf[i] != '':
            ranking[indtoconf[i]] = r_kdd[i]
    sortedrank = sorted(ranking, key=ranking.get, reverse=True)
    for r in sortedrank[0:10]:
        print r, ranking[r]
    print "for top-10 most similar conferences to ICML"
    ranking = {}
    for i in range(0,n):
        if indtoconf[i] != '':
            ranking[indtoconf[i]] = r_icml[i]
    sortedrank = sorted(ranking, key=ranking.get, reverse=True)
    for r in sortedrank[0:10]:
        print r, ranking[r]
    print "for top-10 most similar conferences to KDD and ICML"
    ranking = {}
    for i in range(0,n):
        if indtoconf[i] != '':
            ranking[indtoconf[i]] = r_both[i]
    sortedrank = sorted(ranking, key=ranking.get, reverse=True)
    for r in sortedrank[0:10]:
        print r, ranking[r]
    
