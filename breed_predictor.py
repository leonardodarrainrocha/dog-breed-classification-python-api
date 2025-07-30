import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from werkzeug.datastructures import FileStorage

from breed_dictionary import breeds


__evaluation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def __get_model():
    model_aux = resnet50(weights=ResNet50_Weights.DEFAULT)
    features = model_aux.fc.in_features
    model_aux.fc = torch.nn.Linear(features, 133)
    return model_aux


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'model/model_transfer.pt'
model = __get_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)


def get_prediction(file: FileStorage):
    img = Image.open(file.stream)
    img_t = __evaluation_transforms(img)
    input_tensor = img_t.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        percentages = torch.softmax(outputs.squeeze(), 0)
        maximum_item = torch.max(percentages, 0)
        breed_number = maximum_item.indices.item()
        breed_name = breeds[breed_number]
        final_percentage = int(maximum_item.values.item() * 10000) / 100

    return {"breed_number": breed_number, "breed_name": breed_name, "final_percentage": final_percentage}



