from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
np.random.seed(1234)

def svm_classify(X, label, split_ratios, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor for SVM
    """
    train_size = int(len(X)*split_ratios[0])
    val_size = int(len(X)*split_ratios[1])

    train_data, valid_data, test_data = X[0:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    train_label, valid_label, test_label = label[0:train_size], label[train_size:train_size + val_size], label[train_size + val_size:]

    print('training SVM...')
    clf = svm.SVC(C=C, kernel='linear')
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(train_data)
    train_acc = accuracy_score(train_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)
    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)

    return [train_acc, valid_acc, test_acc]

data = np.random.random((500,10))
label = np.random.randint(0,2,(500,))

split_ratios = [0.6, 0.2, 0.2]
svm_C = 0.1
[train_acc, valid_acc, test_acc] = svm_classify(data, label, split_ratios, svm_C)

print("Train Acc:", train_acc, " Validation Acc:", valid_acc, " Test Acc:", test_acc)