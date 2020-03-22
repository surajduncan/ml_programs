# For a given set of training data examples stored in a .CSV file, 
# implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all Hypotheses consistent with the training examples.




import random
import csv
def g_o(n):
    return ("?",)*n
def s_o(n):
    return ("0",)*n
def more_general(h1,h2):
    more_general_parts=[]
    for x,y in zip(h1,h2):
        mg=x=="?" or (x!="0" and (x==y or y=="0"))
        more_general_parts.append(mg)
    return all(more_general_parts)
    
def fulfills(example,hypothesis):
    return more_general(hypothesis,example)

def min_generalization(h,x):
    h_new=list(h)
    for i in range(len(h)):
        if not fulfills(x[i:i+1],h[i:i+1]):
            h_new[i]="?" if h[i]!="0" else x[i]
    return [tuple(h_new)]
min_generalization(h=('0','0','sunny'),x=('rainy','windy','cloudy'))

def min_specialization(h,domains,x):
    results=[]
    for i in range(len(h)):
        if h[i]=="?":
            for val in domains[i]:
                if x[i]!=val:
                    h_new=h[:i]+(val,)+h[i+1:]
                    results.append(h_new)
        elif h[i]!="0":
            h_new=h[:i]+('0',)+h[i+1:]
            results.append(h_new)
    return results
min_specialization(h=('?','x'),domains=[['a','b','c'],['x','y']],x=('b','x'))
with open('candidate.csv') as csvFile:
    examples=[tuple(line) for line in csv.reader(csvFile)]

def get_domain(examples):
    d=[set() for i in examples[0]]
    for x in examples:
        for i,xi in enumerate(x):
            d[i].add(xi)
    return [list(sorted(x)) for x in d]
get_domain(examples)

def candidate_elimination(examples):
    domains=get_domain(examples)[:-1]
    G=set([g_o(len(domains))])
    S=set([s_o(len(domains))])
    i=0
    print("\nG[{0}]".format(i),G)
    print("\nS[{0}]".format(i),S)
    for xcx in examples:
        i=i+1
        x,cx=xcx[:-1],xcx[-1]
        if cx=='Y':
            G={g for g in G if fulfills(x,g) }
            S=generalize_S(x,G,S)
        else:
            S={s for s in S if not fulfills(x,s)}
            G=specialize_G(x,domains,G,S)
        print("\nG[{0}]:".format(i),G)
        print("\nS[{0}]:".format(i),S)
    return 

def generalize_S(x,G,S):
    s_prev=list(S)
    for s in s_prev:
        if s not in S:
            continue
        if not fulfills(x,s):
            S.remove(s)
            splus=min_generalization(s,x)
            S.update([h for h in splus if any([more_general(g,h) for g in G])])
            S.difference_update([h for h in S if any([more_general(h,h1) for h1 in S if h!=h1])])
    return S

def specialize_G(x,domains,G,S):
    g_prev=list(G)
    for g in g_prev:
        if g not in G:
            continue
        if fulfills(x,g):
            G.remove(g)
            gminus=min_specialization(g,domains,x)
            G.update([h for h in gminus if any([more_general(h,s) for s in S])])
            G.difference_update([h for h in G if any([more_general(g1,h) for g1 in G if h!=g1])])
    return G
candidate_elimination(examples)
