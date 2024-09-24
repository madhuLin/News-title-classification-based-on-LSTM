class word2Sequence():  #建造辭典
  # self
  UNK_TAG = 'UNK' #未出現
  PAD_TAG = "PAD" #填充

  UNK = 0
  PAD = 1
  def __init__(self):
    self.dictW2 = {
      self.UNK_TAG : self.UNK,
      self.PAD_TAG : self.PAD
    }
    self.count = {} #統計詞頻
  def fit(self, sentence):
    """
    store sentence in dictionary
    :param sentence[word1, word2, word3, ...]
    """
    for word in sentence:
      self.count[word] = self.count.get(word, 0)+1 #count.get(word, 0) 如果get不到為 0
  def build_vocab(self, min_count=5, max_count=None, max_feature=None):
    """
    build vocad
    :param min_count: 最小出現次數
    :param max_coint: 最小出現次數
    :param max_feature: 最大詞彙數量
    """
    if min != None:
      self.count = {word:value for word, value in self.count.items() if value >= min_count}
    if max_count != None:
      self.count = {word: value for word, value in self.count.items() if value < max_count} 
    # if max_feature != None: #如果特徵數量大於 max_feature 取最多出現的詞
    #   print(len(self.count))
    #   temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_feature]
    #   self.count = dict(temp)    
    for word in self.count:
      self.dictW2[word] = len(self.dictW2) #運用長度大於下標特性
    self.inverse_dict = dict(zip(self.dictW2.values(), self.dictW2.keys()))
  
  #把句子轉換為序列
  def transform(self, sentence, max_len=None):
    """
    把句子轉換為序列
    :param sentence
    :param max_len
    """
    if max_len != None:
      if max_len > len(sentence):
        sentence += [self.UNK_TAG] * (max_len - len(sentence))
      if max_len < len(sentence):
        sentence = sentence[:max_len]

    return [ self.dictW2.get(word, self.UNK) for word in sentence]

  #將序列轉換為句子
  def inverse_transform(self, indexs):
    return [ self.inverse_dict.get(idx, self.UNK_TAG) for idx in indexs]
  
  def __len__(self):
    return len(self.dictW2)


# import os
# from data_preprocessing import tokenlize
# import pickle

# # if __name__ == '__name__':
#   #將數據讀出來以訓練辭典
# print("sdfsdf")
# ws = word2Sequence()
# data_path = r"c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\data"

# temp_file_path = [os.path.join(data_path, i) for i in os.listdir(data_path)]
# # print(temp_file_path)
# for file_path in temp_file_path:
#   if(file_path == r"c:\程式\Python\AI2022F_基於LSTM之新聞標題分類\data/class.txt"): continue
#   # "c:\程式\Python\AI2022F_基於LSTM之新聞標題分類/models/ws_News.pkl"
#   print(file_path)
#   sentence = tokenlize(open(file_path, encoding="utf-8").read())
#   ws.fit(sentence)
# ws.build_vocab(min_count=10, max_feature=4000)
# pickle.dump(ws, open(r"c:\程式\Python\AI2022F_基於LSTM之新聞標題分類/models/ws_News.pkl", 'wb'))
# print(len(ws.dictW2))
  