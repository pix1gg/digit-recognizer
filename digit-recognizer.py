#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted = [int(x) for x in rf.predict(test)]

    savetxt('Data/rf_submission.csv', predicted, delimiter=',', fmt='%d')

    gb = GradientBoostingClassifier();
    gb.fit(train, target);
    predicted = [int(x) for x in gb.predict(test)]
    savetxt('Data/gb_submission.csv', predicted, delimiter=',', fmt='%d')

if __name__=="__main__":
    main()

