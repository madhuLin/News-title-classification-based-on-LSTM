# import pickle
from word_sequence import word2Sequence


config = {
      "resume" : False,
      'saveModel': True, #是否儲存model
      "batch_size" : 256,  #每批data大小
      'lr' : 0.0001,  #學習速度
      'n_epochs' : 1, #訓練輪數
      'max_len' : 32, #一條數據幾個字 
      'num_layers' : 2, #lstm層數
      'hidden_size' : 128, #隱蔵神經元
      'bidirectional' : True, #雙向LSTM
      'dropout' : 0.4, #丟棄幾%神經元
      'embedding_dim' : 200,  #將w轉成多少長度的   3D matrix m*n*w 

      'save_path' : r"c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models/IMDB_model_LSTM_News.pt",
      'wsPath' : "c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models\ws_News.pkl"
}
# ws = word2Sequence()
# if __name__ == '__main__':
#   ws = pickle.load(open("c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models\ws_News.pkl", 'rb'))
#   # ws = pickle.load(open("c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models\ws_News.pkl", "rb"))

#   print(len(ws.dictW2))

