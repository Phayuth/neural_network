import glob
from dataset_import import dataset_import
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# img_path = glob.glob('./data/test/*/'+'*')
# print(img_path)

data_test = dataset_import('./data/train/')
dataloader_test = DataLoader(dataset = data_test , batch_size = 16, shuffle = True)


for img, label in dataloader_test:
    print(img.shape)
    print(label)
    break

plt.imshow(img[2])
plt.show()

