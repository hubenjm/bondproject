import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import extract
import visualize

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def main():
	d = extract.get_data_simple()
	price = d.pop('price')
	d = d.dropna()
	column_names = d.columns

	bins = np.array([ 0, 80, 90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112.5, 115, 120, 150])
	price = np.digitize(price, bins)

	#scale variables
	d = StandardScaler().fit_transform(d)

	#split into test and training parts
	d_train, d_test, p_train, p_test = train_test_split(d, price, test_size=0.20, random_state=13)

	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
		 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
		 "Naive Bayes", "QDA"]

	classifiers = [
	    KNeighborsClassifier(5),
	    SVC(kernel="linear", C=0.025),
	    SVC(gamma=2, C=1),
	    DecisionTreeClassifier(max_depth=None),
	    RandomForestClassifier(max_depth=None, n_estimators=50, max_features=1),
	    MLPClassifier(alpha=1),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis()]

	for name, clf in zip(names, classifiers):
		clf.fit(d_train, p_train)
		p_pred = clf.predict(d_test)
        	score = clf.score(d_test, p_test)
		print(name + ": number correct predictions/total samples = {}".format(score))
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(p_test, p_pred)
		plt.figure()
		visualize.plot_confusion_matrix(cnf_matrix, classes=bins, title='Confusion matrix, without normalization')
		plt.show()


if __name__ == "__main__":
	main()
