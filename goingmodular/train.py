import os 
import torch
from torchvision import transforms
from goingmodular import data_setup,engine,model_builder,utils
from timeit import default_timer as timer

NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

train_dir="data/pizza_steak_sushi/train"

test_dir="data/pizza_steak_sushi/test"

device="cuda" if torch.cuda.is_available() else "gpu"

data_transform=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
    
])

train_dataloader,test_dataloader,class_names=data_setup.create_dataloaders(train_dir=train_dir,
                                                                          test_dir=test_dir,
                                                                          transforms=data_transform,
                                                                          bath_size=BATCH_SIZE)

model=model_builder.TinyVGG(input_shape=3,hidden_units=HIDDEN_UNITS,output_shape=len(class_names)).to(device)

loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),
                          lr=LEARNING_RATE)



start_time=timer()
engine.train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=NUM_EPOCHS,
            device=device)

end_time=timer()
print(f"total training time for model_1:{end_time-start_time}seconds")

utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_MODULAR_SCRIPT")
