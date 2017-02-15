import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import extract
import visualize
from sklearn import tree
import pydotplus 

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
	#d = extract.get_data_simple()
	s0 = pd.read_csv("price_change_data.csv", usecols = ['yield', 'tradetype',
		'state', 'coupon', 'maturity', 'issuetype', 'issuesource', 'rtg',
		'dtradedate', 'dprice', 'holdtime'])

	#index = s0.pop(0)

	#transform text features
	#s_name_features = extract.build_name_features(s0)
	s_state_features = extract.build_state_features(s0, num_states = 15)
	s_other_features = extract.build_other_text_features(s0)
	s = pd.concat([s0.drop(['state', 'issuetype', 'issuesource', 'tradetype'], axis = 1), s_state_features, s_other_features], axis = 1)
	s = s.dropna()
	s = s[s.holdtime < 6]
	
	dprice = s.pop('dprice')
	dprice_binary = dprice.apply(lambda x: x > 0).astype(np.int)

	print s.columns.tolist(), len(s.columns.tolist())

	#scale variables
	#s = StandardScaler().fit_transform(s)

	#split into test and training parts
	s_train, s_test, dp_train, dp_test = train_test_split(s, dprice_binary, test_size=0.20, random_state=13)

#	I = s_test.holdtime <= 5
#	s_test = s_test[I]
#	dp_test = dp_test[I]

	#visualize some characteristics of the bonds which lead to positive gains
	#print s0[dprice > 0].state.value_counts()



#	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
#		 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#		 "Naive Bayes", "QDA"]
	names = ["Decision Tree", "Random Forest", "Naive Bayes"]

#	classifiers = [
#	    KNeighborsClassifier(5),
#	    SVC(kernel="linear", C=0.025),
#	    SVC(gamma=2, C=1),
#	    DecisionTreeClassifier(max_depth=None),
#	    RandomForestClassifier(max_depth=None, n_estimators=50, max_features=1),
#	    MLPClassifier(alpha=1),
#	    AdaBoostClassifier(),
#	    GaussianNB(),
#	    QuadraticDiscriminantAnalysis()]
	classifiers = [DecisionTreeClassifier(max_depth=None, criterion = 'entropy'),
		RandomForestClassifier(max_depth=None, n_estimators=50, max_features=1),
		GaussianNB()]

	for name, clf in zip(names, classifiers):
		clf = clf.fit(s_train, dp_train)
		if name == "Decision Tree":
			with open("iris.dot", 'w') as f:
				f = tree.export_graphviz(clf, out_file=f)

		dp_pred = clf.predict(s_test)
        	score = clf.score(s_test, dp_test)
		print(name + ": number correct predictions/total samples = {}".format(score))
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(dp_test, dp_pred)
		plt.figure()
		visualize.plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix, without normalization')
		plt.show()


if __name__ == "__main__":
	main()
