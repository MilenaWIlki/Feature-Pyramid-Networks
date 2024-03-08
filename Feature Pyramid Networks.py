import torch
import torchvision

def detect_objects_fpn(image_path, model_name):
    model = torchvision.models.detection.__dict__[model_name](pretrained=True)
    model.eval()
    image = Image.open(image_path)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = transform(image)
    with torch.no_grad():
        prediction = model([image])
    return prediction

# Example usage:
image_path = "image.jpg"
model_name = "fasterrcnn_resnet50_fpn"
objects = detect_objects_fpn(image_path, model_name)
print("Detected objects:", objects)
