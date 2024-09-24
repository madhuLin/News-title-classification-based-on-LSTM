# 介紹

## 基於LSTM之新聞標題分類
### 目標 : 利用RNN的LSTM模型預測新聞標題分類
### 來源 : THUCTC(THU Chinese Text Classification)
### 數據蒐集 : 由THUCTC數據集中取20萬筆數據
### 功能說明 : 基於RNN中Long Short-Term Memory (LSTM)的方法來建立一個能夠辨識新聞標題的分類器。透過採用LSTM網路架構，可以有效地分析新聞標題的語意，並在最後使用線性模型分類新聞類別


# 環境
### Python 3.8
### torch 1.13.0
### tqdm  
### pickle


# 數據集介紹
從[THUCNews](http://thuctc.thunlp.org/)中抽取了20萬條新聞標題，已上傳至github，文本長度在20到30之間。一共10個類別，每類2萬條。數據以字為單位輸入模型。

類別：財經、房產、股票、教育、科技、社會、時政、體育、遊戲、娛樂。

數據集劃分：

數據集|數據量
--|--
訓練集|18萬
驗證集|1萬
測試集|1萬


# 更換自己的數據集
 - 按照我數據集的格式來格式化你的中文數據集。


# 編碼
### coding: utf-8 


# 使用說明
在myLib中設定完自己的路徑後
直接執行mian.py就可以開始訓練自己的模型
```
python main.py
```

# 程式說明
### word_sequence
    讀取資料夾內的新聞文的詞彙用來建立詞典，並可以將句子轉換成序列，也可以將序列轉換成句子。


### data_preprocessing.py 裡面包含tokenlize 、News_Dataset、get_dataloader
    tokenlize : 可以將一篇文章中的文字，變成訓練需要的格式 
    News_Dataset : 繼承自torch.utils.data.Dataset它可以把數據集轉換成 PyTorch可以使用的格式，從而可以使用PyTorch進行深度學習。
    get_dataloader : 它可以按需加載數據，並且可以對數據進行批處理以提高訓練效率。


### model.py  
    IMDBModel : 用於構建神經網路，主要的功能是利用LSTM捕捉時間序列中長期依賴性的模式，再用線性模型進行分類

### myLib.py 
    在myLib中可以調整訓練參數

### train.py 裡面包含train、dev
    train : 用來訓練模型，以及記錄loss和準確率
    dev : 用來驗證模型的準確率
### test.py 
    test : 用來測試模型的準確率

### main.py



## 參考
https://github.com/996icu/996.ICU/blob/master/LICENSE
https://github.com/SpringMagnolia/NLPnote/blob/master/1.3%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/1.3.4%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%AE%9E%E7%8E%B0%E6%83%85%E6%84%9F%E5%88%86%E7%B1%BB.md

http://thuctc.thunlp.org/
https://www.youtube.com/watch?v=ExXA05i8DEQ&ab_channel=Hung-yiLee
https://www.youtube.com/results?search_query=hung-yi+lee+bert
https://bravey.github.io/2020-05-12-%E4%BD%BF%E7%94%A8LSTM+Pytorch%E5%AF%B9%E7%94%B5%E5%BD%B1%E8%AF%84%E8%AE%BA%E8%BF%9B%E8%A1%8C%E6%83%85%E6%84%9F%E5%88%86%E7%B1%BB.html
