import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import StepLR

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.tanh(x)
        output = self.fc2(x)

        return output


def train(batch_size, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % batch_size == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)

            output = model(image)
            test_loss = F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


transform =  transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

cwd = os.getcwd()
train_folder = cwd+"/data/train/"
test_folder = cwd+"/data/test/"
models_folder = cwd+"/models/"

if not os.path.isdir(train_folder):
    os.makedirs(train_folder)

if not os.path.isdir(test_folder):
    os.makedirs(test_folder)

train_dataset = datasets.MNIST(train_folder, download=True, train=True, transform=transform)
test_dataset = datasets.MNIST(test_folder, download=True, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

epochs = 15
batch_size = 64

for epoch in range(epochs):
    train(batch_size, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

    if epoch == epochs - 1:
        if not os.path.isdir(models_folder):
            os.makedirs(models_folder)

        torch.save(model.state_dict(), models_folder+"mnist_lenet5.pt")