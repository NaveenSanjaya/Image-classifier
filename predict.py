import argparse
import json
import numpy as np
from PIL import Image
import torch


def get_input():
    parser = argparse.ArgumentParser(description="Training the Neural Network")
    
    parser.add_argument('--image', dest='image',
                        help='Path to image')
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='Checkpoint of the trained network')
    parser.add_argument('--top_k', dest='top_k', type=int,
                        help='Number of most likely classes')
    parser.add_argument('--catogey_names', dest='catogey_names', action='store',
                        help='Names of the catogeries in .json format')
    parser.add_argument('--gpu', dest='gpu', action='store',
                        help='Use "gpu" for training')
    
    return parser.parse_args()

def load_checkpoint(filepath='checkpoint.pth'):
          
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
        
    for param in model.parameters():
        param.requires_grad = False 
    print(model)           
    print("Done: Loading the checkpoint")
    return model

def process_image(image):
    
    # Loading the image
    image = Image.open(image)
    width, height = image.size
    
    # Resizing the image where shortest side is 256px, and keeping the aspect-ratio
    ratio = min(width, height) / 256
    new_width = int(width / ratio)
    new_height = int(height / ratio)
    image = image.resize((new_width, new_height))
    
    # Cropping the image
    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert the PIL image in to Numpy array
    np_image = np.array(image)
    np_image = np_image / 255
    
    # Normalize the color channels
    means = np.array([[0.485, 0.456, 0.406]])
    stds = np.array([[0.229, 0.224, 0.225]])
    np_image = (np_image - means) / stds
    
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, cat_to_name, gpu, topk=5):
    
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    if gpu == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    image, model = image.to(device), model.to(device)

    model.eval()
    with torch.no_grad():
        
        logps = model.forward(image)
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        probs, labels = probs.cpu().numpy(), labels.cpu().numpy()
        
        class_to_idx = model.class_to_idx
        idx_to_class = {i: j for j, i in class_to_idx.items()}

        top_classes = [idx_to_class[i] for i in labels[0]]
        class_names = [cat_to_name[i] for i in top_classes]

        return probs[0], top_classes, class_names

def main():

    args = get_input()

    with open(args.catogey_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    probs, classes, class_names = predict(args.image, model, cat_to_name, args.gpu, args.top_k)
    print(probs)
    print(classes)
    print(class_names)

if __name__ == '__main__': main()

# Sample input
# python predict.py --image flowers/test/54/image_05402.jpg --checkpoint checkpoint/checkpoint.pth --top_k 5 --catogey_names cat_to_name.json --gpu gpu

