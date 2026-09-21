import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from datapipeline import dataloader


model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Initializing training")

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    total_epoch_loss = 0.0

    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = cost(output, labels)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value
        total_epoch_loss += loss_value

        if i % 10 == 9:
            print(f"Epoch: {epoch + 1}, batch: {i + 1}, Loss: {running_loss / 10:.5f}")
            running_loss = 0.0

    epoch_loss = total_epoch_loss / len(dataloader)
    print(f'--- End of Epoch {epoch + 1} Avg Loss: {epoch_loss:.5f} ---')

    torch.save(model.state_dict(), "leaflens.pth")
    print("Model Saved")

print("Training Finished Big Dawg")