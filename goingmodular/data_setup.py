
import os
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

#how many gpu core is working
NUM_WORKERS=os.cpu_count()


def create_dataloaders(
    train_dir:str,
    test_dir:str,
    train_transforms:transforms.Compose,
    test_transforms:transforms.Compose,
    batch_size:int,
    num_workers:int=NUM_WORKERS
):
    #select all image implement selected transform and turn into tensor 
    train_data=datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data=datasets.ImageFolder(test_dir,transform=test_transforms)
    
    #ImageFolder class have a variable about folder names under the selected path
    class_names=train_data.classes
    
    
    train_dataloader=DataLoader(
        train_data,
        batch_size=batch_size, #how many image should select in each iter
        shuffle=True, #mix the data
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader=DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader,test_dataloader,class_names
    
