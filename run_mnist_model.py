import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import StepLR
from lenet5 import LeNet5


class Experiment(object):
    def __init__(self, model, num_epochs, learning_rate, out_folder):
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.out_folder = out_folder
        self.train_folder = self.out_folder + "/data/train/"
        self.test_folder = self.out_folder + "/data/test/"
        self.models_folder = self.out_folder + "/models/"

        self.gamma = 0.7
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, batch_size, model, device, train_loader, optimizer, epoch):
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

    def test(self, model, device, test_loader):
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

    def get_and_make_train_test_data(self):
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

        if not os.path.isdir(self.train_folder):
            os.makedirs(self.train_folder)

        if not os.path.isdir(self.test_folder):
            os.makedirs(self.test_folder)

        train_dataset = datasets.MNIST(self.train_folder, download=True, train=True, transform=transform)
        test_dataset = datasets.MNIST(self.test_folder, download=True, train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        return train_loader, test_loader

    def run_experiment(self):

        train_loader, test_loader = self.get_and_make_train_test_data()
        model = self.model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)

        for epoch in range(self.num_epochs):
            self.train(self.batch_size, self.model, self.device, train_loader, optimizer, epoch)
            self.test(self.model, self.device, test_loader)
            scheduler.step()

            if epoch == self.num_epochs - 1:
                if not os.path.isdir(self.models_folder):
                    os.makedirs(self.models_folder)

                torch.save(model.state_dict(), self.models_folder + f"mnist_{self.model.name}.pt")

if __name__ == "__main__":
    exp1 = Experiment(model=LeNet5(), num_epochs=15, learning_rate=0.01, out_folder=os.getcwd())
    exp1.run_experiment()