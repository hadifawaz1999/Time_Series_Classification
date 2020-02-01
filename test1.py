from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
np.random.seed(1)

iris = sklearn.datasets.load_iris()
#print(iris)

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))
print(train[0])
for i in range(train[0].size):
    print(train[0][i])
#print(labels_train[0])
#print(test[0],test[0].shape)
#print(labels_test[0])

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)

##print(exp.as_list())

exp.as_pyplot_figure()
##plt.show()