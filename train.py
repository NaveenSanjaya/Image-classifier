import argparse
import os
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def get_input():
    parser = argparse.ArgumentParser(description="Training the Neural Network")
    
    parser.add_argument('--dir', dest='dir', action='store',
                        help='path to folder of images')
    parser.add_argument('--arch', dest='arch', type=str, action='store', default='vgg16',
                        help='Select Network architecture: "alexnet" or "vgg16" or "densenet"')
    parser.add_argument('--save_dir', dest='save_dir', action='store',
                        help='Path where directory shoould be saved')
    parser.add_argument('--learning_rate', dest='learning_rate', action='store',type=float, default=0.001,
                        help='Learning rate of the network')
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, action='store',
                        help='Number of hidden units')
    parser.add_argument('--epochs', dest='epochs', type=int, action='store', default=1,
                        help='Number of hidden epochs')
    parser.add_argument('--gpu', dest='gpu', action='store', default='gpu',
                        help='Use "gpu" for training')
    
    return parser.parse_args()

def train_transformer(data_dir):

    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    print("Done: Transforming tain_data")
    return train_data

def test_transformer(data_dir):
      
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(data_dir, transform=test_transforms)
    print("Done: Transforming test_data")
    return test_data

def dataloader(dataset, shuffle=False):
    if shuffle == False:
        return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=64)
    
def network_builder(architecture, hidden_units):
    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                        ('inputs', nn.Linear(25088, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.2)),
                        ('hidden_layer', nn.Linear(hidden_units, 512)),
                        ('relu', nn.ReLU()),
                        ('hidden_layer', nn.Linear(512, 256)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.2)), 
                        ('hidden_layer', nn.Linear(256, 102)), # Output_size=102
                        ('output', nn.LogSoftmax(dim=1))]))
            
    elif architecture == "alexnet":
        model = models.alexnet(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                        ('inputs', nn.Linear(9216, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.2)),
                        ('hidden_layer', nn.Linear(hidden_units, 256)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.2)), 
                        ('hidden_layer', nn.Linear(256, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))

    elif architecture == "densenet":
        model = models.densenet121(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(hidden_units, 256),
                            nn.ReLU(),
                            nn.Linear(256, 95),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(95, 102),
                            nn.LogSoftmax(dim=1))

    model.classifier = classifier
    print("Done: Building the network")
    print(model)
    return model

def train_network(model, epochs, learning_rate, gpu, trainloader, validloader):
    if gpu == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)                
                        valid_loss += batch_loss.item()
                        
                        # Calculating accuracy of the valid set
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.."
                    f"Train loss: {running_loss/print_every:.3f}.."
                    f"Valid loss: {valid_loss/len(validloader):.3f}.."
                    f"Valid accuracy: {valid_accuracy/len(validloader):.3f}")
                
                running_loss = 0

                model.train()
    print("Done: Training the Network")
    return model, criterion

def validation(model, gpu, criterion, testloader):
    if gpu == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    test_loss = 0
    test_accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculating accuracy of the test set
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.."
        f"Test accuracy: {test_accuracy/len(testloader):.3f}")
    
    print("Done: Validating results")

    running_loss = 0

def save_network(model, hidden_units, learning_rate, epochs, train_data, save_dir):
    checkpoint = {'model': model,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx
                }
    
    os.makedirs(save_dir, exist_ok=True)  
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    print("Done: Saving the network")

def main():

    args = get_input()

    data_dir = args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)

    trainloader = dataloader(train_data, shuffle=True)
    validloader = dataloader(valid_data)
    testloader =dataloader(test_data)

    model = network_builder(args.arch, args.hidden_units)

    trained_network, criterion = train_network(model, args.epochs, args.learning_rate, args.gpu, trainloader, validloader)

    validation(trained_network, args.gpu, criterion, testloader)

    save_network(model, args.hidden_units, args.learning_rate, args.epochs, train_data, args.save_dir)

if __name__ == '__main__': main()

# Sample input
# python train.py --dir flowers --arch vgg16 --save_dir checkpoint --learning_rate 0.002 --hidden_units 1024 --epochs 3 --gpu gpu
# python train.py --dir flowers --arch alexnet --save_dir checkpoint --learning_rate 0.002 --hidden_units 512 --epochs 3 --gpu gpu
# python train.py --dir flowers --arch densenet --save_dir checkpoint --learning_rate 0.002 --hidden_units 512 --epochs 1 --gpu gpu
