import pandas as pd
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from huggingface_hub import login

# Login to Hugging Face Hub
login(token="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")

# Set device to CUDA:0
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# Load the processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")

# Define prompts
prompts = {
    "Recognised Food Item": "Which is the food item in the image?",
    "Recipe generated": "Generate a recipe for the dish shown in the image.",
    "List of ingredients predicted": "Provide the list of ingredients.",
    "Step by step instructions generated": "Provide step-by-step instructions."
}

# Load the CSV file
csv_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/30_Dishes - newfilefiltered.csv'
data = pd.read_csv(csv_path)

# Function to process each prompt
def process_image(image_path, prompt):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, prompt=prompt, return_tensors="pt").to(device, torch.float16)
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True).strip()

# Process each image and save responses
responses = []
for index, row in data.iterrows():
    image_name = row['Image_Name']
    image_path = f'/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/Images (finaL)/{image_name}.jpg'
    
    response_dict = {'Image_Name': image_name}
    
    for key, prompt in prompts.items():
        print(f"Processing Image: {image_path} | Prompt: {prompt}")
        response = process_image(image_path, prompt)
        print(f"Response: {response[:100]}...")  # Print first 100 characters
        response_dict[key] = response
    
    responses.append(response_dict)
    
    # Optional: Save after each image in case of interruption
    temp_df = pd.DataFrame(responses)
    temp_df.to_csv('responses_blip2.csv', index=False)
    print(f"Saved responses for {image_name}")

# Convert responses to a DataFrame and save to CSV
response_df = pd.DataFrame(responses)
response_df.to_csv('responses_blip2.csv', index=False)

print("Processing complete. Responses saved to: responses_blip2.csv")