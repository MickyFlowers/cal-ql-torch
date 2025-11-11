from urllib.request import urlopen

import timm
import torch
from PIL import Image


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
print("Image size:", img.size)
model = timm.create_model('resnet18.a1_in1k', pretrained=True, features_only=True, out_indices=(4,))
model = model.eval()
print("Model created.")

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
print(transforms)
transforms_image = transforms(img).unsqueeze(0)  # unsqueeze single image into batch of 1
transforms_image = extend_and_repeat(transforms_image, 1,5)
output = model(transforms_image)  # unsqueeze single image into batch of 1
print("Output feature map shape:", output[0].shape)
# top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
