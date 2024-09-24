import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
from tqdm import tqdm

from word_sequence import word2Sequence 
# from myLib import ws
import pickle

# import pickle
# ws = pickle.load(open("c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models\ws_News.pkl", 'rb'))
from myLib import config

def tokenlize(content): #文字格式
  """
  content: a string of text
  """
  content = re.sub("<.*?>"," ", content)
  content = content.replace(" ","")
  fileters =['：',"\t","\n", "\x97","\x96","<br>","<br/>","#","\.","$","%","&","/","-","\(","\)","\"","\?",'>','<','^']
  content = re.sub("|".join(fileters), " ", content)
  content = content.replace(" ","")
  tokens = [i for i in list(content)]
  return tokens

# ws = word2Sequence()
# with open("c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\models\ws_News.pkl", 'rb') as f:
#   ws = pickle.load(f)


"""# **準備數據集dataset**"""

class News_Dataset(Dataset):
  ''' 用於加載和預處理 News 數據集的數據集 '''
  def __init__(self, mode) -> None:
    super(News_Dataset).__init__()
    data_path = r'data'
    # if mode == 'test':
    #   data_path = path
    # else:
    # 把所有文件名放入列表
    # temp_file_path = [os.path.join(data_path,'pos'),os.path.join(data_path,'neg')]

    self.total_text = [] #所有文件路徑

    for name in os.listdir(data_path): 
      if(name.split('.')[0] == mode):  #判斷現在是甚麼模式
        file_path = os.path.join(data_path, name)

    #將data append 進list
    with open(file_path, 'r',encoding="utf-8") as f:
      for line in tqdm(f):
        lin = line.strip() #去除首尾空格
        if not lin:
          continue
        self.total_text.append(lin) 

  #取得資料
  def __getitem__(self, index):
    lin = self.total_text[index]
    #獲取label
    content, label = lin.split('\t') 
    label = int(label)
    tokens = tokenlize(content)  
    return tokens, label 

  #獲取長度
  def __len__(self):
    return len(self.total_text)


#重寫 collate_fn 
def collate_fn(bacth):
  content, label = list(zip(*bacth))
  ws = pickle.load(open(config['wsPath'], 'rb'))
  content = [ws.transform(i, max_len=32) for i in content]
  content = torch.LongTensor(content)
  label = torch.LongTensor(label)
  return content, label


def get_dataloader(mode, batch_size):
  ''' 生成數據集，然後放入數據加載器。 '''
  imdb_dataset = News_Dataset(mode=mode)
  data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=(mode=='train'), collate_fn=collate_fn)
  return data_loader
