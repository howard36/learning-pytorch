import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_ds = datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True)

test_ds = datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        download=True)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.linear = nn.Linear(10*4*4, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

def train_one_epoch(model, train_dl, loss_fn, optim):
    model.train()
    for x, y in train_dl:
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

def evaluate(model, dl, loss_fn):
    total_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            correct += torch.sum(torch.eq(torch.argmax(logits, dim=1), y)).item()
    return total_loss / len(dl), 100 * correct / len(dl.dataset)

def train(model, train_dl, test_dl, loss_fn, optim, num_epochs=5):
    print(f"Initial: {evaluate(model, test_dl, loss_fn)}")
    for i in range(num_epochs):
        train_one_epoch(model, train_dl, loss_fn, optim)
        train_loss, train_acc = evaluate(model, train_dl, loss_fn)
        test_loss, test_acc = evaluate(model, test_dl, loss_fn)
        print(f"Epoch {i}: Train loss {train_loss:.4f}, Train acc {train_acc:>4.2f}, Test loss {test_loss:.4f}, Test acc {test_acc:>4.2f}")

loss_fn = torch.nn.CrossEntropyLoss()
model = Net()
#model.load_state_dict(torch.load("model.pth"))
optim = torch.optim.Adam(model.parameters())


train(model, train_dl, test_dl, loss_fn, optim)
torch.save(model.state_dict(), "model.pth")
