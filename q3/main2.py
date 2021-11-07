import math
import time
import numpy as np



#accuracy, precision, recall and f1 score calculations
def scores(labels_test,predicted_labels,time):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for n in range(0,len(labels_test)):
        if predicted_labels[n] == 1 and labels_test[n] == 1:
            TP +=1
        if predicted_labels[n] == 0 and labels_test[n] == 0:
            TN +=1
        if predicted_labels[n] == 1 and labels_test[n] == 0:
            FP +=1
        if predicted_labels[n] == 0 and labels_test[n] == 1:
            FN +=1
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(recall * precision) / (recall + precision)
    print("accuracy = %.3f " % accuracy)
    print("precision = %.3f " % precision)
    print("recall = %.3f " % recall)
    print("f1 = %.3f " % f1)
    print("time passed = %.2f seconds\n" % (time))




#count how many spam and ham sms and calculates p(0) and p(1),
#also returns the training labels
def ham_spam_counter(label):    
    f2 = open(label)
    spam = 0
    ham = 0
    
    first_row = 0
    labels = []
    for lines in f2:
        if first_row == 0:
            first_row +=1
        else:
            tokens = lines.split(",")
            token = tokens[1].strip("\n")
            labels.append(int(token))
            if int(token) ==0:
                ham +=1
            elif int(token) == 1:
                spam +=1
    
    f2.close()
    
    P0 = ham/ (spam+ham)
    P1 = spam/(spam+ham)


    return ham, spam, P0, P1, labels

#calculate how many times words are found in spam and ham sms
def probs(data,labels):
    f1 = open(data)
    
    
    i = 0
    j = 0
    first_row = 0
    word_spam =  [0] * 3458
    word_ham =  [0] * 3458
    
    word_count =  [0] * 3458
    total_word = 0
    
    
    for lines in f1:
        if first_row == 0:
            first_row +=1
        else:
            tokens = lines.split(",")
            j = 0
            first_column = 0
            for token in tokens:
                if first_column==0:
                    first_column +=1
                else:
                    word_count[j] += int(token)
                    total_word += int(token)
                    if labels[i] == 0:
                        if int(token) != 0:
                            word_ham[j] += 1
                    elif labels[i] == 1:
                        if int(token) != 0:
                            word_spam[j] += 1
                    
                    j+=1
            i+=1    
    
    f1.close()
    return word_spam, word_ham, word_count, total_word


#calculate the probability of each P(xj|y=1) and P(xj|y=0)
def probs_calculate(word_spam,word_ham,spam,ham,P0,P1):

    
    
    P_word_spam = np.array(word_spam) / spam
    P_word_ham= np.array(word_ham) / ham

    
    P_spam = math.log(P1)
    P_ham = math.log(P0)

    return P_word_ham,P_word_spam,P_ham,P_spam


#calculates the mutual information
def mutual_information(data,labels,word_count,P_word_spam,P1,P_word_ham,P0):
    f1 = open(data)
    
    
    
    i = 0
    j = 0
    first_row = 0
    mutual_info = [0] * 3458
    
    
    
    for lines in f1:
        if first_row == 0:
            first_row +=1
        else:
            tokens = lines.split(",")
            j = 0
            first_column = 0
            for token in tokens:
                if first_column==0:
                    first_column +=1
                else:
                    if labels[i] == 1:
                        if word_count[j] == 0:
                            mutual_info[j]= 0
                        else:
                            conditional_e = P_word_spam[j]/((word_count[j])*P1)
                            if conditional_e:
                                mutual_info[j] += P_word_spam[j] *math.log(conditional_e,2)
                            else:
                                mutual_info[j] += P_word_spam[j]
                    elif labels[i] == 0 :
                        if word_count[j] == 0:
                            mutual_info[j]= 0
                        else:
                            conditional_e =  P_word_ham[j]/((word_count[j])*P0)
                            if conditional_e:
                                mutual_info[j] += P_word_ham[j] * math.log(conditional_e,2)
                            else:
                                mutual_info[j] += P_word_ham[j]
                    
                    j+=1
            i+=1
    
    
    f1.close()
    
    return mutual_info

#finds the features with highest mutual information
def best_features(mutual_info):
    return (sorted(range(len(mutual_info)), key=lambda k: mutual_info[k]))



#load test labels
def load_label(test_label):
    f4 = open(test_label)
    
    first_row = 0
    labels_test = []
    for lines in f4:
        if first_row == 0:
            first_row +=1
        else:
            tokens = lines.split(",")
            token = tokens[1].strip("\n")
            labels_test.append(int(token))
    f4.close()
    return labels_test




#prediction without feature selection
def predict_with_all_features(test_data,P_ham,P_spam,P_word_ham,P_word_spam):
    start = time.time()
    f3 = open(test_data)
    
    
    
    first_row = 0
    ham_count = 0
    spam_count = 0
    predicted_labels = []
    for lines in f3:
        if first_row == 0:
            first_row +=1
        else:
            tokens = lines.split(",")
            j = 0
            first_column = 0
            is_Ham = P_ham
            is_Spam = P_spam
            for token in tokens:
                if first_column==0:
                    first_column +=1
                else:   
                    if (int(bool(int(token)))):
                        if P_word_ham[j]!= 0:
                            is_Ham += math.log(P_word_ham[j])
                        if P_word_spam[j] != 0:
                            is_Spam += math.log(P_word_spam[j])

                    else:  
                        is_Ham += math.log(1-P_word_ham[j])
                        is_Spam += math.log(1-P_word_spam[j])
    
                    j +=1
        
      
    
            if is_Ham >= is_Spam:
                ham_count +=1
                predicted_labels.append(0)
            else:   
                spam_count +=1
                predicted_labels.append(1)
    
    
    
    f3.close()
    end=time.time()
    return predicted_labels,(end-start)





#predict with the n number of features using the mutual information
def predict_with_features(test_data,feature_size,P_ham,P_spam,P_word_ham,P_word_spam,best_features):
    start= time.time()
    f3 = open(test_data)
    first_row = 0
    ham_count = 0
    spam_count = 0
    predicted_labels = []
    for lines in f3:
        if first_row == 0:
            first_row +=1
        else:
            tokens = lines.split(",")
            count = 1
            first_column = 0
            is_Ham = P_ham
            is_Spam = P_spam
            for j in best_features:
                if count == feature_size:
                    break
                if first_column==0:
                    first_column +=1
                else:   
                    if (int(bool(int(tokens[j])))):
                        if P_word_ham[j]!= 0:
                            is_Ham += math.log(P_word_ham[j])
                        if P_word_spam[j] != 0:
                            is_Spam += math.log(P_word_spam[j])

                    else:  
                        is_Ham += math.log(1-P_word_ham[j])
                        is_Spam += math.log(1-P_word_spam[j])
    
                    count +=1
        
    
    
            if is_Ham >= is_Spam:
                ham_count +=1
                predicted_labels.append(0)
            else:   
                spam_count +=1
                predicted_labels.append(1)
    f3.close()
    end = time.time()
    return predicted_labels, (end-start)


data = "sms_train_features.csv"
label = "sms_train_labels.csv"
test_data = "sms_test_features.csv"
test_label = "sms_test_labels.csv"


ham, spam, P0, P1, labels = ham_spam_counter(label)
word_spam, word_ham, word_count, total_word = probs(data,labels)
P_word_ham,P_word_spam,P_ham,P_spam = probs_calculate(word_spam, word_ham, spam, ham, P0, P1)
mutual_info = mutual_information(data,labels,word_count,P_word_spam,P1,P_word_ham,P0)
best_features = best_features(mutual_info)
labels_test = load_label(test_label)



predicted_labels, time1 = predict_with_all_features(test_data,P_ham,P_spam,P_word_ham,P_word_spam)
scores(labels_test,predicted_labels, time1)

predicted_labels0 , time1 = predict_with_features(test_data,100,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels0, time1)

predicted_labels1 , time1 = predict_with_features(test_data,200,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels1, time1)

predicted_labels2 , time1 = predict_with_features(test_data,300,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels2, time1)

predicted_labels3 , time1 = predict_with_features(test_data,400,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels3, time1)

predicted_labels4 , time1 = predict_with_features(test_data,500,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels4, time1)

predicted_labels5 , time1 = predict_with_features(test_data,600,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels5, time1)
"""
predicted_labels6 , time1 = predict_with_features(test_data,700,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels6, time1)

predicted_labels7 , time1 = predict_with_features(test_data,800,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels7, time1)

predicted_labels8 , time1 = predict_with_features(test_data,900,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels8, time1)

predicted_labels9 , time1 = predict_with_features(test_data,1000,P_ham,P_spam,P_word_ham,P_word_spam,best_features)
scores(labels_test,predicted_labels9, time1)

"""







