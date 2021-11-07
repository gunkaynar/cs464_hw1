import math

#load data and train the multinomial model
def train():
    data = "sms_train_features.csv"
    label = "sms_train_labels.csv"


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
    
    
    f1 = open(data)
    
    i = 0
    j = 0
    first_row = 0
    word_count = [0] * 3458
    word_spam =  [0] * 3458
    word_ham =  [0] * 3458
    
    
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
                    if labels[i] == 0:
                        word_ham[j] += int(token)
                    elif labels[i] == 1:
                        word_spam[j] += int(token)
                    
                    j+=1
            i+=1    
    
    f1.close()
    
    
    P_word_ham = []
    P_word_spam = []
    P_spam = math.log(P1)
    P_ham = math.log(P0)
    
    for word in range(0,len(word_count)):
        if word_ham[word] == 0:
            P_word_ham.append(float('-inf'))
        else:
            P_word_ham.append(math.log((word_ham[word])))
        if word_spam[word] == 0:
            P_word_spam.append(float('-inf'))
        else:
            P_word_spam.append(math.log((word_spam[word])))

    return P_ham, P_spam, P_word_ham, P_word_spam




#test labels are load into array
def test_labels():
    test_label = "sms_test_labels.csv"
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

#prediction based on trained multinomial model
def predict():
    test_data = "sms_test_features.csv"
    
    
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
                    if float(token):
                        is_Ham +=(P_word_ham[j] * float(token)) 
                        is_Spam +=(P_word_spam[j] * float(token))
                    j +=1
    
            if is_Ham >= is_Spam:
                ham_count +=1
                predicted_labels.append(0)
            else:   
                spam_count +=1
                predicted_labels.append(1)
    
    
    
    f3.close()
    return predicted_labels

#accuracy, precision, recall and f1 score calculation
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
    print("accuracy = %.3f " % accuracy)
    print("precision = %.3f " % precision)
    print("recall = %.3f " % recall)
    print("f1 = %.3f " % f1)
    return TP, TN, FP, FN


P_ham, P_spam, P_word_ham, P_word_spam = train()
predicted_labels = predict()

labels_test =  test_labels()


TP, TN, FP, FN = scores(labels_test,predicted_labels)






















