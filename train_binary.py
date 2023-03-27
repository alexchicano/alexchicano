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
dataloaders, mtd, IDs, id_dict = get_data(data_transforms, batch_size, num_workers, interval = [[0, 6], [11, 14]], binarize = True) 

# Define your model and optimizer
model = Net3c2d_bin()
model_name = 'bin_3c2d_6-11_3.pth'
model.load_state_dict(torch.load('../models/bin_3c2d_6-11_2.pth'))
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)


# Setting up the training
SAVE_PATH = f'../models/{model_name}.pth'
print_every = 1
num_epochs = 50
criterion = nn.MSELoss()

# Early stopping criteria
patience = 10

if __name__ == '__main__':
    print('Training the model...')
    # train the model
    loss_train, loss_val = train_model(model, dataloaders, criterion, optimizer, num_epochs, patience, print_every, SAVE_PATH)

    # save the losses
    save_losses(model_name, loss_train, loss_val)

    # compute the predictions on the test set
    test_df = save_test_predictions(model, dataloaders, mtd, csv_path = f'../results/predictions/preds_{model_name}.csv')

    # save the parameters
    test_loss = get_test_loss(model, dataloaders, criterion)
    best_train_loss, best_val_loss = [min(loss_train), min(loss_val)]

    accuracy = get_accuracy(test_df)
    print(f'Accuracy: {accuracy}')

    # calculate the sensitivity and specificity
    sensitivity, specificity = get_sensitivity_specificity(test_df)


    # Read the csv file called "model_settings.csv" in the "models" folder
    csv_path = os.path.join('..', 'models', f'{model_name}.csv')


    # if the file does not exist, create it
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns = ['model_name', 'batch_size', 'best_epoch', 'optimizer', 'lr', 'criterion', 'patience',
                                    'best_train_loss', 'best_val_loss', 'test_loss', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc'])
        df.to_csv(csv_path, index = True)
        
    df = pd.read_csv(csv_path, index_col = 0)
    df.loc[0] = [model_name, batch_size, np.argmax(loss_train), 'Adam', lr, 'MSE', patience, best_train_loss, best_val_loss, test_loss, accuracy, sensitivity, specificity, 'Unknown', 'Unknown']
    df.to_csv(csv_path, index = True)
