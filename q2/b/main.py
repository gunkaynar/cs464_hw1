import pandas as pd
import numpy as np
import copy
import time

#load dataset with pandas and put them in numpy arrays
def load():
    train = "diabetes_train_features.csv"
    train_label = "diabetes_train_labels.csv"
    test = "diabetes_test_features.csv"
    test_label = "diabetes_test_labels.csv"
    
    
    df = pd.read_csv(train)
    df = df.drop('Unnamed: 0', axis=1)
    df_label = pd.read_csv(train_label)
    df_test = pd.read_csv(test)
    df_test = df_test.drop('Unnamed: 0', axis=1)
    df_test_label = pd.read_csv(test_label)
    
    
    features_np_array = df.to_numpy()
    labels_np_array = df_label.to_numpy()
    test_features_np_array = df_test.to_numpy()
    test_labels_np_array = df_test_label.to_numpy()
    return features_np_array,labels_np_array,test_features_np_array,test_labels_np_array



#euclidian distance calculation
def euclidian_distance(array1, array2):
    return np.sqrt(np.sum((array1-array2)**2, axis=1))

#k-nearest neigbors 
def knn(test_features_np_array, features_np_array, k=9, return_distance=False):
    distances = []
    neigbor = []
    points = [euclidian_distance(features_test, features_np_array) for features_test in test_features_np_array]
    for point in points:
        enum_neigbor = enumerate(point)
        sorted_neigbor = sorted(enum_neigbor, key=lambda x: x[1])[:k]
        ind_list = [tup[0] for tup in sorted_neigbor]
        dist_list = [tup[1] for tup in sorted_neigbor]
        distances.append(dist_list)
        neigbor.append(ind_list)
    if return_distance:
        return np.array(distances), np.array(neigbor)
    return np.array(neigbor)


#prediction based on neigbors
def predict(test_features_np_array,features_np_array,labels_np_array):
    predicted_label = []
    neigbors = (knn(test_features_np_array, features_np_array))
    for sample in neigbors:
        prediction = 0
        for neigbor in sample:
            prediction += labels_np_array[:,1][neigbor]
        if prediction > 4.5:
            predicted_label.append(1)
        else:
            predicted_label.append(0)
    return np.array(predicted_label)

#accuracy, precision, recall and f1 score calculation
def scores(labels, predicted_labels,val,prin=0):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for n in range(0,len(labels)):
        if predicted_labels[n] == 1 and labels[:,1][n]== 1:
            TP +=1
        if predicted_labels[n] == 0 and labels[:,1][n]== 0:
            TN +=1
        if predicted_labels[n] == 1 and labels[:,1][n] == 0:
            FP +=1
        if predicted_labels[n] == 0 and labels[:,1][n] == 1:
            FN +=1
    if prin:
        print("TP = %i " % TP)
        print("TN = %i " % TN)
        print("FP = %i" % FP)
        print("FN = %i" % FN)
        
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(recall * precision) / (recall + precision)
    if prin:
        print("accuracy = %.3f " % accuracy)
        print("precision = %.3f " % precision)
        print("recall = %.3f " % recall)
        print("f1 = %.3f " % f1)
    
    if val == "accuracy":
        return accuracy
    elif val == "precision":
        return precision
    elif val == "recall":
        return recall
    elif val == "f1":
        return f1
   


#backward elimination
def bacward_elimination(test_features_np_array,features_np_array,labels_np_array,test_labels_np_array,metric):
    names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    change = 1
    eliminated_features = []
    while change:
        predicted_labels = predict(test_features_np_array,features_np_array,labels_np_array)
        a = (scores(test_labels_np_array,predicted_labels,metric))
        copy_np_test = copy.deepcopy(test_features_np_array)
        copy_np_train = copy.deepcopy(features_np_array)
        eliminated = []
        for i in range(0,len(copy_np_test[0,:])-1):
            copy_np_test = np.delete(copy_np_test, i, axis=1)
            copy_np_train = np.delete(copy_np_train, i, axis=1)
            predicted_labels = predict(copy_np_test,copy_np_train,labels_np_array)
            eliminated.append(scores(test_labels_np_array, predicted_labels,metric))
            copy_np_test = copy.deepcopy(test_features_np_array)
            copy_np_train = copy.deepcopy(features_np_array)
        if max(eliminated) >= a:
            features_np_array = np.delete(features_np_array, eliminated.index(max(eliminated)), axis=1)
            test_features_np_array = np.delete(test_features_np_array, eliminated.index(max(eliminated)), axis=1)
            start = time.time()
            predicted_labels = predict(test_features_np_array,features_np_array,labels_np_array)
            end = time.time()
            print("time passed = %.2f seconds" % (end-start))
            a = (scores(test_labels_np_array,predicted_labels,metric,1))
            eliminated_features.append(names[eliminated.index(max(eliminated))])
            names.pop(eliminated.index(max(eliminated)))
        else:
            change = 0
    return eliminated_features

features_np_array,labels_np_array,test_features_np_array,test_labels_np_array = load()


print("all features")
start = time.time()
predicted_labels = predict(test_features_np_array,features_np_array,labels_np_array)
end = time.time()
print("time passed = %.2f seconds" % (end-start))
(scores(test_labels_np_array,predicted_labels,"f1",1))
eliminated_features0 = bacward_elimination(test_features_np_array,features_np_array,labels_np_array,test_labels_np_array,"accuracy")
print(eliminated_features0)

"""
print("precision: ")
eliminated_features1 = bacward_elimination(test_features_np_array,features_np_array,labels_np_array,test_labels_np_array,"precision")
print("recall: ")
eliminated_features2 = bacward_elimination(test_features_np_array,features_np_array,labels_np_array,test_labels_np_array,"recall")
print("f1: ")
eliminated_features3 = bacward_elimination(test_features_np_array,features_np_array,labels_np_array,test_labels_np_array,"f1")
"""

