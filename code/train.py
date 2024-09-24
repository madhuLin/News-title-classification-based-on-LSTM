
import pickle
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F


def train(train_loader, dev_loader, model, config, optimizer, device):
  n_epochs = config['n_epochs']
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
  loss_train = []      # for recording training loss
  acc_record = {'train': [], 'dev': []}
  max_correct = 0.0
  for epoch in range(n_epochs):
    model.train()
    tatol_loss = 0
    train_correct = 0
    train_num = 0
    for x, y in tqdm(train_loader):
      x, y = x.to(device), y.to(device) # move data to device (cpu/cuda)
      optimizer.zero_grad()   # set gradient to zero
      pred = model.forward(x)
      loss = model.cal_loss(pred, y)
      loss.backward()   # forward pass (compute output)
      optimizer.step()  # update model with optimizer
      tatol_loss += loss.detach().cpu().item()
      values, labels = torch.max(pred, dim=-1, keepdim=False)
      train_num += len(x.detach().cpu())
      train_correct += sum(labels.detach().cpu().numpy() == y.detach().cpu().numpy()) 
      loss_train.append(tatol_loss / len(train_loader.dataset))
    acc_record['train'].append(train_correct/train_num)
      # correct += sum(labels.detach().cpu().numpy() == y.detach().cpu().numpy())
    # print('epoch = {:4d}, loss = {:.4f}'.format(epoch+1, tatol_loss))
    print('epoch = {:4d}, loss = {:.4f} acc = {:.4f})'.format(epoch + 1, tatol_loss, train_correct/train_num ))
    
    # After each epoch, test your model on the validation (development) set.
    dev_correct = dev(dev_loader, model, device)
    acc_record['dev'].append(dev_correct)

    #防止過擬合train data
    if dev_correct > max_correct:
      # Save model if your model improved
      max_correct = dev_correct
      print('Saving model (epoch = {:4d}, loss = {:.4f} acc = {:.4f})'
          .format(epoch + 1, tatol_loss, max_correct))
        
      if config['saveModel']:
        torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
        # torch.save(optimizer.state_dict(), r"/content/drive/MyDrive/Colab Notebooks/Machine Learning/model/IMDB_optimizer_LSTM_News.pt")
    
  print('Finished training')
  
  return (loss_train, acc_record)


  #驗證準確率
def dev(dev_loader, model, device):
  model.eval()
  dev_loss = 0
  correct=0
  for x, y in tqdm(dev_loader):
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
      pred = model.forward(x) 
      mse_loss = F.nll_loss(pred, y, reduction='sum')
    dev_loss += mse_loss.detach().cpu().item()
    values, labels = torch.max(pred, dim=-1, keepdim=False)
    correct += sum(labels.detach().cpu().numpy() == y.detach().cpu().numpy())
  return correct /len(dev_loader.dataset)
  print('Accuracy{:.2f}  Loss:{:.2f}'.format(correct/len(dev_loader.dataset), dev_loss))