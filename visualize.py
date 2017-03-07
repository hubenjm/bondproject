import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
						  normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues, fontsize = 16, tick_fontsize = 14):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	image = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.set_title(title, fontsize = fontsize)
	cb = plt.colorbar(image, ax=ax)
	cb.ax.tick_params(labelsize=tick_fontsize) 
	tick_marks = np.arange(len(classes))
	ax.set_xticks(tick_marks)
	ax.set_xticklabels(classes, rotation=45, fontsize = tick_fontsize)
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(classes, fontsize = tick_fontsize)
	ax.minorticks_off()

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, cm[i, j],
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="black", fontsize = fontsize)

	ax.set_ylabel('True label', fontsize = fontsize)
	ax.set_xlabel('Predicted label', fontsize = fontsize)
	return fig
