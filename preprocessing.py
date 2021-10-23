import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

### read the original data file and split it into data and labels
df = pd.read_csv("Cleaned-Data.csv", header=0).drop(["Country"], axis=1)
labels = df.filter(["Severity_Mild", "Severity_Moderate", "Severity_None", "Severity_Severe"], axis=1)
data = df.drop(labels, axis = 1)

### change the labels to 0, 1, 2, 3 from the original 
classes = []
for i in range(labels.shape[0]):
    
    mild = labels.iloc[i]["Severity_Mild"]
    moderate = labels.iloc[i]["Severity_Moderate"]
    healthy = labels.iloc[i]["Severity_None"] 
    severe = labels.iloc[i]["Severity_Severe"]
    
    if mild == 1 and moderate == 0 and healthy == 0 and severe == 0:
        classes.append(1)
    if mild == 0 and moderate == 1 and healthy == 0 and severe == 0:
        classes.append(2)
    if mild == 0 and moderate == 0 and healthy == 1 and severe == 0:
        classes.append(0)
    if mild == 0 and moderate == 0 and healthy == 0 and severe == 1:
        classes.append(3)

classes = pd.DataFrame({"class": classes})
result = pd.concat([data, classes], axis = 1)

### split data into train and test. We choose 20% as test and 80% as train
train, test = train_test_split(result, test_size=0.2, random_state=42, shuffle=True)

if __name__ == '__main__':
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)