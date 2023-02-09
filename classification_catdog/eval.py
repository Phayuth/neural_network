# Force to use GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# Import torch
import torch
from torch.utils.data import DataLoader

# Import custom dataset class
from dataset_import import dataset_import

# Import plot
import matplotlib.pyplot as plt

# Create Dataset Class from custom image in folder for Testing
dataset_test  = dataset_import('./dataset/test/')

# Load Data into Dataloader for Testing
dataloader_test  = DataLoader(dataset = dataset_test, batch_size = 2000)#, shuffle = True)

model = torch.load("./weight/modelsave.pt")
model.eval()

# Availble Class
classes_list = [
    "Cat",
    "Dog"]

index = 22

for imgt, targt in dataloader_test:
    pred = model(imgt.to("cuda"))
    predicted = classes_list[pred[index].argmax().item()]
    actual    = classes_list[targt[index].item()]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    break

plt.imshow(imgt[index])
plt.show()