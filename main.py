#pip install pythainlp==3.1.1
#pip install gensim

# Documentation : https://pythainlp.github.io/docs/3.1/

import codecs,json,pickle
from sklearn.linear_model import ElasticNet
from pythainlp.word_vector import WordVector
import numpy as np

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def load_dataset(fname):
    fp = codecs.open(fname,encoding="utf-8") 
    data = json.load(fp)
    print(len(data))
    return data

def get_word_vector():
    wv = WordVector()
    model = wv.get_model()
    a = model["บ้าน"]
    print(a)

def create_input_and_label():
    wv = WordVector()

    SIZE = -1

    train_data = load_dataset("./splits/train.json")
    X = []
    Y = []
    i = 0

    if SIZE > 0:
        train_set = train_data[0:SIZE] + train_data[-SIZE:]
    else:
        train_set = train_data

    for sample in train_set:
        print(i)
        text = sample[0].replace(" ","")
        x = wv.sentence_vectorizer(text,use_mean=True)
        X.append(x[0])
        if sample[1] == "depression":
            Y.append(0.9)
        else:
            Y.append(0.1)
        i += 1
    print(Y)
    return np.array(X),np.array(Y)

def build_model(x,y):
    print("Fitting....")
    regr = ElasticNet(alpha=0.001,max_iter=10000,random_state=0,selection="random")

    scaler.fit(x)
    new_x = scaler.transform(x)

    regr.fit(new_x,y)
    print("Fitting completed")
    return scaler,regr 


def train_model():
    #load_dataset("./splits/train.json")
    X,Y = create_input_and_label()
    scaler, model = build_model(X,Y)
    print(model.coef_)

    filename = 'finalized_model.sav'
    pickle.dump([scaler,model], open(filename, 'wb'))

def test_model():
    filename = 'finalized_model.sav'
    scaler,model = pickle.load(open(filename,"rb"))
    print(scaler,model.coef_)

    test_data = load_dataset("./splits/test.json")
    
    wv = WordVector()

    count = 0

    text = "ดีใจจังเลยจะได้กลับบ้านแล้ว"
    i = 0
    for sample in test_data:
        text = sample[0].replace(" ","")
        print(i, text)
        x = wv.sentence_vectorizer(text,use_mean=True)
        x = scaler.transform(x)
        out = model.predict(x)
        ans = "depression" if out[0] > 0.5 else "no_depression"
        print("Label : ",sample[1])
        print("Ans   : ",ans)

        if sample[1] == ans:
            count += 1

        print("---")
        i += 1

    print("ACC : ", count / len(test_data))

if __name__ == "__main__":
    test_model()