from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor  # modern replacement for ViTFeatureExtractor

class MiniImageNetFusionDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        self.resnet_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = self.data[idx]["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor_resnet = self.resnet_tf(image)  # (3, 224, 224)
        vit_inputs = self.vit_processor(images=image, return_tensors="pt")
        return img_tensor_resnet, vit_inputs["pixel_values"].squeeze(0), label
