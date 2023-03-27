from metadata import *
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import os


# Create a custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, id_dict, transform=None, interval = [], binarize = False):
        self.transform = transform
        self.image_paths = [id_dict[id]['preprocessed'] for id in id_dict.keys()]
        self.scores = [id_dict[id]['score'] for id in id_dict.keys()]
        
        # if bins is not empty, then we want to binarize the scores
        if len(interval) != 0 and binarize == True:
            # Binarize the scores:
            for i in range(len(self.scores)):
                # iterate over "scores" and assign "i" to the self.scores that are in the interval[i][0] and interval[i][1] range:
                for j in range(len(interval)):
                    if self.scores[i] >= interval[j][0] and self.scores[i] <= interval[j][1]:
                        self.scores[i] = j
                        break
        # "else" case is non-empty interval and binarize = False, which means we want to keep the scores as they are
        
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = nib.load(img_path).get_fdata()
        label = self.scores[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data(data_transforms, batch_size, num_workers, interval = [], binarize = False):
    
    # Get only a dataset for a specific interval
    mtd = DatasetMetadata( 'ImaGenoma', 'T1_b', interval = interval)
    IDs = mtd.IDs
    id_dict = mtd.id_dict

    # Get two datasets, one for training and one for testing. NOTE: make sure you keep "manual_seed(0)" to get the same split every time
    whole_dataset =  CustomImageDataset(id_dict, transform=data_transforms, interval=interval, binarize=binarize)
    splits = ['train', 'val', 'test']
    
    len_train, len_val = [round(.6 * len(whole_dataset)), round(.2 * len(whole_dataset))]
    len_test = len(whole_dataset) - len_train - len_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        whole_dataset, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(0))

    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in splits}
    
    return dataloaders, mtd, IDs, id_dict


def imshow(img):
    # functions to show an image
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg[:,:,100], cmap='gray')
    plt.show()


def save_test_predictions(net, dataloaders, mtd, csv_path):
    #obtain the batch size from the dataloader
    batch_size = dataloaders['test'].batch_size

    # make a csv file with the predictions for the test set and their labels
    test_df = pd.DataFrame(columns=['ID', 'Predicted', 'Label'])

    # get the indices of the test set in batch_size chunks
    indices = dataloaders['test'].dataset.indices
    indices = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]

    # obtain the predictions for the test set
    for i, data in tqdm(enumerate(dataloaders['test'])):
        batch_indices = indices[i]
        images, labels = data
        inputs = images.unsqueeze(1).double()
        predicted = net(inputs).detach().numpy()
        labels = labels.detach().numpy()
        
        if len(predicted.shape) == 2:
            predicted = np.argmax(predicted, axis=1)
            for j in range(len(predicted)):
                ID = mtd.df.iloc[batch_indices[j]].ID
                test_df.loc[len(test_df)] = [ID, predicted[j], labels[j]]
        else:
            for j in range(len(predicted)):
                ID = mtd.df.iloc[batch_indices[j]].ID
                test_df.loc[len(test_df)] = [ID, round(predicted[j]), round(labels[j])]    
    
    # save the csv file
    test_df.to_csv(csv_path)
    
    return test_df


def get_accuracy(df):
    # This is going to include more metrics in the future, for now it only computes the accuracy
    correct = 0
    for i in range(len(df)):
        if abs(df['Predicted'][i] - df['Label'][i]) < 1:
            correct += 1
    accuracy = correct / len(df)
    return accuracy    


def compute_loss(model, inputs, targets, criterion = nn.MSELoss()):
    
    inputs = inputs.unsqueeze(1).double()
    outputs = model(inputs)
    
    if outputs.shape[1] > 1: # Classification
        _, outputs = torch.max(outputs, 1)
        outputs = outputs.view(-1) # Convert to same size as labels
        
        loss = criterion(outputs.double(), targets)
        loss = Variable(loss, requires_grad = True)
    
    else:
        outputs = outputs.view(-1) # Convert to same size as labels
        loss = criterion(outputs, targets)
        
    return loss

def save_losses(model_name, loss_train, loss_val):
    # Create a "losses.csv" in the results folder, if the csv doesn't exist
    if not os.path.exists('../results/losses.csv'):
        df = pd.DataFrame(columns=['model_name', 'train_loss', 'val_loss'])
        df.to_csv('../results/losses.csv')

    # save the losses as a csv file
    df = pd.read_csv('../results/losses.csv', index_col=0)
    df.loc[len(df)] = [model_name, loss_train, loss_val]
    df.to_csv(f'../results/losses.csv')

def get_test_loss(model, dataloaders, criterion):
    # compute the loss on the test set
    with torch.no_grad():
        test_loss = 0
        for inputs, targets in dataloaders['test']:
            loss = compute_loss(model, inputs, targets, criterion)
            test_loss = test_loss + loss.item()
    return test_loss/len(dataloaders['test'])

def get_sensitivity_specificity(df):
    tp = df[(df['Predicted'] == 1) & (df['Label'] == 1)].shape[0]
    fn = df[(df['Predicted'] == 0) & (df['Label'] == 1)].shape[0]
    tn = df[(df['Predicted'] == 0) & (df['Label'] == 0)].shape[0]
    fp = df[(df['Predicted'] == 1) & (df['Label'] == 0)].shape[0]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def train_model(model, dataloaders, criterion, optimizer, num_epochs, patience, print_every, SAVE_PATH):
    train_losses = []
    val_losses = []
    
    print('Computing initial loss...')
    # start by calculating the loss on the validation set
    with torch.no_grad():
        best_val_loss = 0.0
        for inputs, targets in dataloaders['val']:
            loss = compute_loss(model, inputs, targets)
            best_val_loss += loss.item()
        
    best_val_loss /= len(dataloaders['val'])
    print('Initial validation loss: {:.4f}'.format(best_val_loss))
    
    
    print('Training model...')
    counter = 0
    # Train your model
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            inputs, targets = data
            targets = targets.double()
            loss = compute_loss(model, inputs, targets, criterion)
            loss.backward()
            optimizer.step()
            
            # print statistics
            train_loss += loss.item()
            
            if i % print_every == 0:    # print every "print_every" mini-batches
                print(f'[{epoch + 1}, {i}] loss: {train_loss / (i+1):.3f}')
                
        # store the loss:
        train_losses.append(train_loss / len(dataloaders['train']))
                
        # Calculate validation loss
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in dataloaders['val']:
                loss = compute_loss(model, inputs, targets)
                val_loss += loss.item()
            val_loss /= len(dataloaders['val'])
            # store the loss:
            val_losses.append(val_loss)
            
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), f'{SAVE_PATH}')
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping criteria met, val_loss: {best_val_loss}")
                    break
                
        print('Epoch:', epoch+1 , 'Val Loss:', val_loss, 'Best Loss:' , best_val_loss)
        # Check for early stopping
        if counter >= patience:
            break
        
    print('Finished Training')
    
    return  train_losses, val_losses
    

class Net3c2d(nn.Module):
    # 3 convolutional layers, 2 dense layers
    def __init__(self):
        
        super().__init__()
        self.cl1 = self._conv_layer_set(1, 64)    # 1 input channel, 64 output channels, 3x3 kernel
        self.cl2 = self._conv_layer_set(64, 128)  # 64 input channels, 128 output channels, 3x3 kernel
        self.dropout_cl2 = nn.Dropout3d(p=0.3)    
        self.cl3 = self._conv_layer_set(128, 128) # 128 input channels, 128 output channels, 3x3 kernel
        self.AP = nn.AvgPool3d((2, 2, 2))         # 2x2x2 kernel
        self.fc1 = self._dense_layer_set(3456, 128) # 2**3*128 input channels, 128 output channels
        self.dropout_fc1 = nn.Dropout(p=0.3)
        self.fc2 = self._dense_layer_set(128, 1)
        self.double()
        
    def _dense_layer_set(self, in_c, out_c):
        dense_layer = nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.ReLU()
        )
        return dense_layer
    
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((3, 3, 3)),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer

    def forward(self, x):
        x = self.cl1(x)
        x = self.cl2(x)
        x = self.dropout_cl2(x)
        x = self.cl3(x)
        x = self.AP(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x


class Net3c2d_bin(nn.Module):
    # 3 convolutional layers, 2 dense layers
    def __init__(self):
        
        super().__init__()
        self.cl1 = self._conv_layer_set(1, 64)    # 1 input channel, 64 output channels, 3x3 kernel
        self.cl2 = self._conv_layer_set(64, 128)  # 64 input channels, 128 output channels, 3x3 kernel
        self.dropout_cl2 = nn.Dropout3d(p=0.3)    
        self.cl3 = self._conv_layer_set(128, 128) # 128 input channels, 128 output channels, 3x3 kernel
        self.AP = nn.AvgPool3d((2, 2, 2))         # 2x2x2 kernel
        self.fc1 = self._dense_layer_set(3456, 128) # 2**3*128 input channels, 128 output channels
        self.dropout_fc1 = nn.Dropout(p=0.3)
        self.fc2 = self._dense_layer_set(128, 2)
        self.double()
        
    def _dense_layer_set(self, in_c, out_c):
        dense_layer = nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.ReLU()
        )
        return dense_layer
    
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((3, 3, 3)),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer

    def forward(self, x):
        x = self.cl1(x)
        x = self.cl2(x)
        x = self.dropout_cl2(x)
        x = self.cl3(x)
        x = self.AP(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x