from bert_classifier import *
from torch.nn import CrossEntropyLoss
import torch.optim as optim

device = "cuda:0"

def load_data():

    sentence1 = "วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"
    sentence2 = "ที่ภูเก็ตคนเยอะมาก อาหารก็แพง ไม่น่าเที่ยวแล้ว"

    label1 = "POS"
    label2 = "NEG"

    label2tensor = {"POS" : 0, 
                    "NEG" : 1}
    
    input_tensor = my_batch_tokenizer([sentence1, sentence2])
    input_label = torch.tensor([label2tensor[label1] , label2tensor[label2]])

    return input_tensor, input_label
 

def train(model, lossfn, optimizer, data):

    train_x , train_y = data

    for i in range(0,20):
        optimizer.zero_grad()
        predict = model(train_x.to(device))

        loss = lossfn(predict, train_y.to(device))
        print("LOSS = ", loss.item())
        loss.backward()

        optimizer.step()

def main():
    
    train_x, train_y = load_data()
    model = HoogBERTaClassifier(2)
    model = model.to(device)

    loss = CrossEntropyLoss(reduction="mean")

    optimizer = optim.SGD(model.parameters(), lr = 0.001)
    

    train(model, loss, optimizer, (train_x, train_y))

    predict = model(train_x.to(device))
    predict = predict.argmax(dim=-1)

    print(predict)


if __name__ == "__main__":
    main()

