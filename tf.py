from sklearn import metrics, model_selection
import tensorflow as tf
from tensorflow.contrib import learn

def main(argv):
    iris = learn.datasets.load_iris()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size = .2, random_state = 42)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
    classifier = learn.DNNClassifier(feature_columns = feature_columns, hidden_units = [10,20,10], n_classes=3)
    
    classifier.fit(x_train, y_train, steps=200)

    x_predit = classifier.predict_classes(x_test)
    x_predit = [x for x in x_predit]
    score = metrics.accuracy_score(y_test, x_predit)
    print('Accuracy: {0:f}'.format(score))

main(5)