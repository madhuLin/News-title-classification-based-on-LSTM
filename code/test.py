from tqdm import tqdm
import torch
import torch.nn.functional as F


#測試資料
def test(test_loader, model, device):
  # print(test_loader.dataset[0])
  model.eval()
  test_loss = 0
  correct=0
  for x, y in tqdm(test_loader):
    # print(test_loader.dataset[0])
    with torch.no_grad():
      x, y = x.to(device), y.to(device)
      pred = model.forward(x) 
    test_loss += F.nll_loss(pred, y, reduction='sum')
    
    values, labels = torch.max(pred, dim=-1, keepdim=False)
    # class1 = ['金融','不動產','股票','教育','科學','社會','政治','運動的','遊戲','娛樂']
    # print()
    # print('labels=', end="")
    # print(labels.detach().cpu().numpy())
    # print('target=', end="")
    # print(y.detach().cpu().numpy())
    # print('labels=', end="")
    # print([class1[i] for i in labels.tolist()])
    # print('target=', end="")
    # print([class1[i] for i in y.tolist()])
    correct += sum(labels.detach().cpu().numpy() == y.detach().cpu().numpy())
  test_loss = test_loss / len(test_loader.dataset)
  print('Accuracy {:.2f}   Loss:{:.2f}'.format(correct/len(test_loader.dataset), test_loss))