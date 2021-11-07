    
def load(data,label):
    f1 = open(data)
    ID = []
    Pregnancies = []
    Glucose= []
    BloodPressure= []
    SkinThickness= []
    Insulin= []
    BMI= []
    DiabetesPedigreeFunction= []
    Age= []
    i=0
    for line in f1:
        if i == 0:
            i+=1
            continue
        else:
            tokens = line.split(",")
            ID.append(int(tokens[0]))
            Pregnancies.append(float(tokens[1]))
            Glucose.append(float(tokens[2]))
            BloodPressure.append(float(tokens[3]))
            SkinThickness.append(float(tokens[4]))
            Insulin.append(float(tokens[5]))
            BMI.append(float(tokens[6]))
            DiabetesPedigreeFunction.append(float(tokens[7]))
            Age.append(float(tokens[8].strip("\n")))
    f1.close()
    
    f2 = open(label)
    Diabetes = []
    j=0
    for line in f2:
        if j==0:
            j+=1
            continue
        else:
            tokens = line.split(",")
            Diabetes.append(int(tokens[1].strip("\n")) )
    f2.close()

    return ID,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Diabetes


