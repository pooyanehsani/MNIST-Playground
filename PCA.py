import numpy as np
#in module we calculate the k PCA for the given data
class PCA():
	def __init__(self, k):
		self.component = k
	def __call__(self):
		return self
	def fit_transform(self, x_train, x_test):
		M = np.mean(x_train, axis=0)
		C = x_train - M
		V = np.cov(C.T)
		values, vectors = np.linalg.eig(V)
		eig_pairs = [(np.abs(values[i]), vectors[:, i]) for i in range(len(values))]
		eig_pairs.sort(key=lambda x: x[0], reverse=True)
		trans_matrix = np.empty([784,0])
		for i in range(0,self.component):
			trans_matrix = np.hstack((trans_matrix,eig_pairs[i][1].reshape(784,1)))
		x_train_trans = x_train.dot(trans_matrix)
		x_test_trans = x_test.dot(trans_matrix)
		return (x_train_trans,x_test_trans)
if __name__ == "__main__":
	PCA()