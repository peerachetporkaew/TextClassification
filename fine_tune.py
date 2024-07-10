from bert_classifier import *
from torch.nn import MSELoss
import torch.optim as optim


def load_data():

    sentence1 = "วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"
    sentence2 = "ที่ภูเก็ตคนเยอะมาก อาหารก็แพง ไม่น่าเที่ยวแล้ว"

    label1 = "POS"
    label2 = "NEG"

    label2tensor = {"POS" : [1.0, 0.0], 
                    "NEG" : [0.0,1.0]}
    
    input_tensor = my_batch_tokenizer([sentence1, sentence2])
    input_label = torch.tensor([label2tensor[label1] , label2tensor[label2]])

    return input_tensor, input_label
 

def train(model, lossfn, optimizer, data):

    train_x , train_y = data

    for i in range(0,100):
        optimizer.zero_grad()
        predict = model(train_x)

        loss = lossfn(train_y, predict)
        print("LOSS = ", loss.item())
        loss.backward()

        optimizer.step()

def main():
    
    train_x, train_y = load_data()
    model = HoogBERTaClassifier(2)

    loss = MSELoss()

    optimizer = optim.SGD(model.parameters(), lr = 0.001)

    train(model, loss, optimizer, (train_x, train_y))

    predict = model(train_x)
    predict = predict.argmax(dim=-1)

    print(predict)

    
    # sentence = "วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"
    
    # tokenized_text = my_batch_tokenizer([sentence, sentence, sentence])
    
    # logits = model(tokenized_text)

    # print(logits.shape) # Batch, Dim = 2
    # print(logits)

if __name__ == "__main__":
    main()

