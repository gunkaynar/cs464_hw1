import load_data
import time
import copy

#dictionary hack
def get_key(val,dic):
    for key, value in dic.items():
         if val == value:
             return key




#data is loaded into arrays
train = "diabetes_train_features.csv"
train_label = "diabetes_train_labels.csv"
test = "diabetes_test_features.csv"
test_label = "diabetes_test_labels.csv"


ID,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Diabetes = load_data.load(train,train_label)
ID_test,Pregnancies_test,Glucose_test,BloodPressure_test,SkinThickness_test,Insulin_test,BMI_test,DiabetesPedigreeFunction_test,Age_test,Diabetes_test = load_data.load(test,test_label)

train_data = []
test_data = []


train_data.append(Pregnancies)
train_data.append(Glucose)
train_data.append(BloodPressure)
train_data.append(SkinThickness)
train_data.append(Insulin)
train_data.append(BMI)
train_data.append(DiabetesPedigreeFunction)
train_data.append(Age)
train_label = Diabetes


test_data.append(Pregnancies_test)
test_data.append(Glucose_test)
test_data.append(BloodPressure_test)
test_data.append(SkinThickness_test)
test_data.append(Insulin_test)
test_data.append(BMI_test)
test_data.append(DiabetesPedigreeFunction_test)
test_data.append(Age_test)
test_label = Diabetes_test






#euclidean distance calculation and class prediction
def predict(train_data,test_data,test_type):
    predicted_labels = []
    for testsize in range(0,len(test_data[0])):
        distances = {}
        for size in range(0,len(train_data[0])):
            distance = 0
            for feature in range (0,len(test_data)):
                if test_type == "cv":
                    if testsize == size:
                        continue
                    else:
                        distance += (test_data[feature][testsize]-train_data[feature][size])**2
                elif test_type == "test":
                    distance += (test_data[feature][testsize]-train_data[feature][size])**2
            distances[size] = distance**0.5
            
        neighbours = []
        for k in range(0,8):
            neighbours.append(get_key(sorted(distances.values())[k],distances))
        
        prediction = 0
        for neighbour in neighbours:
            prediction += (train_label[neighbour])
        if prediction > 4.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return predicted_labels
    
    
    

#accuracy, precision, recall, f1 calculations
def scores(labels, predicted_labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for n in range(0,len(labels)):
        if predicted_labels[n] == 1 and labels[n] == 1:
            TP +=1
        if predicted_labels[n] == 0 and labels[n] == 0:
            TN +=1
        if predicted_labels[n] == 1 and labels[n] == 0:
            FP +=1
        if predicted_labels[n] == 0 and labels[n] == 1:
            FN +=1
    
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(recall * precision) / (recall + precision)
    return accuracy, precision, recall, f1, TP, TN, FP, FN


#accuracy, precision, recall, f1 scores calculations for cross-validation
def monitor_scores(train_data,train_label,val):
    start = time.time()
    prediction = predict(train_data,train_data,"cv")
    accuracy,precision,recall,f1, TP, TN, FP, FN = scores(train_label,prediction)
    end = time.time()
    if val == 0:
        print("accuracy = %.3f " % accuracy)
        print("precision = %.3f " % precision)
        print("recall = %.3f " % recall)
        print("f1 = %.3f " % f1)
        print("time passed = %.2f seconds\n" % (end-start))
        print("TP = %i " % TP)
        print("TN = %i " % TN)
        print("FP = %i" % FP)
        print("FN = %i" % FN)  
        return (end-start)
    elif val == "accuracy":
        return accuracy
    elif val == "precision":
        return precision
    elif val == "recall":
        return recall
    elif val == "f1":
        return f1
    
#accuracy, precision, recall, f1 scores calculations for test
def monitor_scores_test(train_data,test_data,test_label,val):
    start = time.time()
    prediction_test = predict(train_data,test_data,"test")
    accuracy,precision,recall,f1, TP, TN, FP, FN = scores(test_label,prediction_test)
    end = time.time()
    if val == 0:
        print("accuracy = %.3f " % accuracy)
        print("precision = %.3f " % precision)
        print("recall = %.3f " % recall)
        print("f1 = %.3f " % f1)
        print("time passed = %.2f seconds\n" % (end-start))
        print("TP = %i " % TP)
        print("TN = %i " % TN)
        print("FP = %i" % FP)
        print("FN = %i" % FN)        
        return (end-start)
    elif val == "accuracy":
        return accuracy
    elif val == "precision":
        return precision
    elif val == "recall":
        return recall
    elif val == "f1":
        return f1
    
    

#backward elimination
def eliminate_features(train_data,train_label,test_data,test_label,metric="accuracy"):
    eliminated = ""
    print("cross-validation with all features: ")
    time_train_first = monitor_scores(train_data,train_label,0)
    print("test with all features: ")
    time_test_first = monitor_scores_test(train_data,test_data,test_label,0)
    names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    while True:
        this_turn = []
        for i in range(0,len(train_data)):
            print(".")
            train_data_temp = copy.deepcopy(train_data)
            first = monitor_scores(train_data_temp,train_label,metric)
            train_data_temp.pop(i)
            second = monitor_scores(train_data_temp,train_label,metric)
            train_data_temp = copy.deepcopy(train_data)
            this_turn.append(second-first)
        if max(this_turn) >= 0:
            train_data.pop(this_turn.index(max(this_turn)))
            test_data.pop(this_turn.index(max(this_turn)))
            print("\nwithout " + names[this_turn.index(max(this_turn))] + ":")
            eliminated += str(names[this_turn.index(max(this_turn))]) + ", "
            names.pop(this_turn.index(max(this_turn)))
            print("cv:\n")
            monitor_scores(train_data,train_label,0)
            print("test:\n")
            monitor_scores_test(train_data,test_data,test_label,0)
        else:
            break
    print("\ncross-validation with eliminated features which are " + eliminated[:-2] + ":")
    time_train_second = monitor_scores(train_data,train_label,0)
    print("\ntest with eliminated features which are " + eliminated[:-2] + ":")
    time_test_second = monitor_scores_test(train_data,test_data,test_label,0)
    
    print("\ntime gained on cross-validation = " + str(round(time_train_first-time_train_second,3)) + " seconds")
    print("\ntime gained on test = " + str(round(time_test_first-time_test_second,3)) + " seconds")

    
eliminate_features(train_data,train_label,test_data,test_label);

























