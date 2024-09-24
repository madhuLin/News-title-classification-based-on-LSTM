

# from lib import model, device
from train import train
from data_preprocessing import get_dataloader
from model import IMDBModel
# from data_preprocessing 
import torch
import matplotlib.pyplot as plt
from test import test
from word_sequence import word2Sequence
from myLib import config 
import pickle


def get_device(): #取得是否有GPU
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu' 
  

if __name__ == '__main__':
    # ws = None
    # sys.path
    device = get_device() 
    print(config)
    # print(len(ws.dictW2))
    model = IMDBModel(config=config)
    # model = model.load_state_dict(torch.load("c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models\IMDB_model_LSTM_News.pt",map_location='cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) 
    
    # model = IMDBModel(config=config)
    train_loader = get_dataloader(mode='train', batch_size=config['batch_size'])
    dev_loader = get_dataloader(mode='dev', batch_size=512)
    test_loader = get_dataloader(mode='test', batch_size=512)

    loss_record = train(train_loader, dev_loader, model, config, optimizer, device)
    

    #畫圖
    train_loss, acc = loss_record 
    plt.figure(figsize=(18,6))
    plt.plot(train_loss, label = "Train loss") #畫loss曲線 
    plt.xlabel("Train loss")
    plt.ylabel("Loss")
    plt.show()

    train_loss, acc = loss_record #rain_loss, acc = loss_record #取
    plt.figure(figsize=(18,6))
    plt.plot(acc['train'], color='red') #訓練集準卻率
    plt.plot(acc['dev'] ,color='blue')  #驗證集準卻率
    plt.xlabel("Validation set accuracy")
    plt.ylabel("training set accuracy")
    plt.show()

    ws = pickle.load(open("c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models\ws_News.pkl", "rb"))

    # print(len(ws.dictW2))

    # print(device, '*'*10)
    model.load_state_dict(torch.load(r"c:\程式\Python\AI2022F_基於LSTM之新聞標題分類/models/IMDB_model_LSTM_News.pt",map_location='cpu'))
    # print(len(test_loader.dataset))
    #測試
    test(test_loader, model, device)




