Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. 
Read the training data from a .CSV file.






import csv
def loadCsv(filename):
    lines=csv.reader(open(filename,"r"))
    dataset=list(lines)
    headers=dataset.pop(0)
    return dataset,headers
def print_hypothesis(h):
    print('<',end=' ')
    for i in range(0,len(h)-1):
        print(h[i],end=' ')
    print('>')
def findS():
    dataset,features=loadCsv('finds.csv')
    rows=len(dataset)
    col=len(dataset[0])
    flag=0
    for x in range(0,rows):
        t=dataset[x]
        if t[-1]=='1' and flag==0:
            flag=1
            h=dataset[x]
        elif t[-1]=='1':
            for y in range(col):
                if h[y]!=t[y]:
                    h[y]='?'
            print("The maximally specific hypothesis for a given training examples")
            print_hypothesis(h)
findS()
