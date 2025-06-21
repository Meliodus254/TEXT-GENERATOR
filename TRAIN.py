import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(784 + 10, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        validity = self.model(x)
        return validity

# Training
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(generator.parameters(), lr=0.0002)
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(30):  # Increase to 30 for better quality
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Train Generator
            noise = torch.randn(batch_size, 100).to(device)
            gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
            gen_imgs = generator(noise, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # Train Discriminator
            real_loss = criterion(discriminator(imgs.to(device), labels.to(device)), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        print(f"Epoch {epoch+1}/10 | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), "generator.pth")

if __name__ == "__main__":
    train()
