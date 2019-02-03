from sklearn import tree
import csv
features = []
labels = []
with open('./iris.csv','r') as csvfile:
    r = csv.reader(csvfile)
    next(r, None)
    for row in r:
        print(row)
        features.append(row[0:4])
        labels.append(row[4])
print(features)
print(labels)

classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)
print(classifier.predict([[7.2,3.2,6,1.8]]))