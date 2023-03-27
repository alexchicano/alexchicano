from metadata import *
from utils_cnn import *
import torchvision.transforms as transforms
import torch.optim as optim


# load MNI image
MNI_PATH = '../data/datasets/_MNI_template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
mni = nib.load(MNI_PATH).get_fdata()
mni_mean, mni_std = [mni.mean(), mni.std()]

center_crop = 200
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mni_mean, mni_std, inplace=True),
    #transforms.CenterCrop(center_crop)
])

# Hyperparameters
batch_size = 4
num_workers = 2

# Get the data structures (delete the interval if not needed)
dataloaders, mtd, IDs, id_dict = get_data(data_transforms, batch_size, num_workers, interval = [[0, 5], [9, 14]])

# Create the network
net = Net3c2d()

# Setting up the training
SAVE_PATH = '../models/trial_model_3.pth'
print_every = 5
num_epochs = 30
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01) # lr = 0.001 was too slow

if __name__ == '__main__':
    train_model(net, dataloaders, criterion, optimizer, num_epochs, print_every, SAVE_PATH)
