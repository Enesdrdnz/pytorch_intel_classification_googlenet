i used the googlenet transfer learning model for intel classification data set in kaggle and i realized somethings i have used also resnet101 model and the resnet101 model is 170mb model 
but the googlenet model is 50mb and the accuries is equal so, model selection is very important, you can do same calculation less than half time or more maybe


## MORE INFO: torchinfo is an library about display models parameters and structures 
when we want restructure classfication layer, we can look at the bottom for flattening because we should know classification layer name 
model.fc = nn.Sequential(nn.Linear(2048, 512),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(512, len(class_names)),
                               nn.LogSoftmax(dim=1)).to(device)
