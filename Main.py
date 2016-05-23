from Common import Common
from LSTM import LSTM
import numpy as np

from sklearn.cross_validation 	 import StratifiedKFold
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier

DS = Common().generate_ucf_50_dataset('frames')
X, y , video_frame_windows= DS

#classifier = KNeighborsClassifier(n_neighbors=10)
classifier = LSTM()
classifier.fit(X,y)
pred = classifier.predict(X)

preds = []
y_test_result = []
acc = 0
for i in range(len(video_frame_windows)):
    preds.append(pred[acc + video_frame_windows[i]-1])
    y_test_result.append(y[acc + video_frame_windows[i]-1])
    acc += video_frame_windows[i]

print preds
print y_test_result

print("precision : ", precision_score(y_test_result, preds))
#print("recall : ", recall_score(y_test_result, preds))
#print("f_score : ", f1_score(y_test_result, preds))
print("accuracy : ", accuracy_score(y_test_result, preds))
#classifier = LSTM()
"""
stf = StratifiedKFold(y, n_folds=10)
for train_index, test_index in stf:
    X_train = []
    X_test  = []
    y_train = []
    y_test  = []

    for x in train_index:
        X_train.append(X[x])
        y_train.append(y[x])

    for x in test_index:
        X_test.append(X[x])
        y_test.append(y[x])

    classifier.fit(X_train,y_train)
    pred = classifier.predict(X_test)

    preds = []
    y_test_result = []
    for i in range(len(video_frame_windows)):
        preds.append(pred[video_frame_windows[i]])
        y_test_result.append(y_test[video_frame_windows[i]])

    print("precision : ", precision_score(y_test_result, preds))
    #print("recall : ", recall_score(y_test, pred))
    #print("f_score : ", f1_score(y_test, pred))
    print("accuracy : ", accuracy_score(y_test_result, preds))
"""