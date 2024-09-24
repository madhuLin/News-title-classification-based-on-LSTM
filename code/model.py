import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import pickle
# from data_preprocessing import ws


class IMDBModel(nn.Module):
  def __init__(self, config):
    super(IMDBModel, self).__init__()
    ws = pickle.load(open(config['wsPath'], 'rb'))
    
    #nn.Embedding() num_embedding:所有文本詞彙Index的數量    embedding_dim:一個詞應該轉換成多少維度的向量
    self.embedding = nn.Embedding(len(ws),config['embedding_dim']) #將詞由數字轉換為矩陣 [batch_size, max_len, 100]
    self.net = nn.Sequential(
        # nn.Dropout(config['dropout']-0.1),
        # nn.ReLU(),
        nn.Linear(config['hidden_size']*2,10)
      # nn.Linear(config['hidden_size']*2,64),
      # nn.ReLU(),
      # nn.Dropout(config['dropout']-0.1),
      # nn.Linear(64,10)
    )
    #LSTM   # input_size: embedding第三維常度,  hidden_size:隱藏神經元   num_layers:幾層LSTM   batch_first: batch_size是否為第一維   bidirectional:是否為雙向  dropout:防止過擬合
    self.lstm = nn.LSTM(input_size=config['embedding_dim'], hidden_size=config['hidden_size'], num_layers=config['num_layers'], batch_first=True, bidirectional=config['bidirectional'], dropout=config['dropout'])
    
    self.criterion = nn.NLLLoss()  #取 log 並將正確答案的機率值加總取平均


  def forward(self, input):
    x = self.embedding(input) #shape:[batch_size, max_len, 300]

    x, (h_n, c_h) = self.lstm(x) #h_n 隱藏狀態  c_n 細胞狀態
    #x: [batch_size, max_len, 2*hidden_size ]   h_n = [num_layers*bidirectional, batch_size, hidden_size, ]
    #獲取兩個方向最後一個output 進行content
    output_fw = h_n[-2,:,:] #正向最後一次輸出
    output_bw = h_n[-1,:,:] #反向最後一次輸出
    output = torch.cat([output_fw, output_bw], dim=-1) #[batch_size, hidden_size*2]
    x = self.net(output)
    return F.log_softmax(x, dim=-1)  #使得每一個元素的範圍都在(0,1)之間，並且所有元素的和為1


  def cal_loss(self,pred, target):
    # values, labels = torch.max(pred, 1)
    # # return self.criterion(pred, target)
    # labels = torch.Tensor(labels)
    # print(pred, target)
    return self.criterion(pred, target)