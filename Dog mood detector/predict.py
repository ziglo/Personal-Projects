import torch
from torchvision import transforms
from PIL import Image
import os
from models.mood_classifier import MoodClassifier

class DogMoodPredictor:
    def __init__(self, model_path):
        # 1. Load the checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # Map to CPU for safety
        
        # 2. Setup the model with the correct number of classes from the checkpoint
        num_classes = checkpoint['num_classes']
        self.model = MoodClassifier(num_classes=num_classes).model
        
        # 3. Load the weights specifically (Fixes the crash)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 4. Load the label mapping (Fixes the "output is a number" issue)
        self.idx_to_label = checkpoint['idx_to_label']
        
        # 5. Define the same transforms (with normalization!)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            
        # Convert the integer index (e.g., 0) to the string label (e.g., "Happy")
        predicted_index = predicted.item()
        return self.idx_to_label[predicted_index]

if __name__ == "__main__":
    # Correct model path
    model_path = 'dog_mood_regNet.pth'
    
    # Check if file exists before crashing
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did you run train.py?")
    else:
        predictor = DogMoodPredictor(model_path)
        
        image_path = input("Enter the path to the dog image: ")
        
        # Clean up path string (removes quotes if you drag-and-drop file in terminal)
        image_path = image_path.strip('"').strip("'")
        
        if os.path.exists(image_path):
            mood = predictor.predict(image_path)
            print(f"The predicted mood of the dog is: {mood}")
        else:
            print("Error: Image file not found.")