import numpy as np
import pandas as pd
import visualize
import extract
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

def main():
	d = extract.get_data()
	d = extract.clean_data(d)
	d_name_features = extract.build_name_features(d)
	d_state = extract.build_state_features(d, 15)
	d_other = extract.build_other_text_features(d)
	
	d = d.drop(['state', 'name', 'issuetype', 'issuesource', 'tradetype'], axis = 1)
	d = pd.concat([d, d_state, d_other, d_name_features], axis = 1)	

	column_names = d.columns
	price = d.pop('price')

	bins = np.array([ 0, 80, 90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,112.5,115, 120, 150])
	#bins = np.array([ 0, 80, 90, 95, 98, 100, 102, 105, 110, 120, 150])
	price = np.digitize(price, bins)
	#scale variables
	d = StandardScaler().fit_transform(d)

	#split into test and training parts
	d_train, d_test, p_train, p_test = train_test_split(d, price, test_size=0.20, random_state=33)

	names = ["Decision Tree", "Random Forest", "AdaBoost",
		 "Naive Bayes", "Gradient Boosted Tree", "Logistic Regression"]

	classifiers = [
	    DecisionTreeClassifier(max_depth=None),
	    RandomForestClassifier(max_depth=None, n_estimators=200, max_features=10),
	    AdaBoostClassifier(),
	    GaussianNB(), GradientBoostingClassifier(n_estimators = 200, max_depth = None, learning_rate = 0.1, verbose = 1, max_features = 'sqrt'),
		LogisticRegression(multi_class = 'ovr')]

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
