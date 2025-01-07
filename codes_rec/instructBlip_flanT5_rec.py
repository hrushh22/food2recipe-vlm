import pandas as pd
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch

# Set device to CUDA:0
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# Load the model and processor
model_id = "Salesforce/instructblip-flan-t5-xl"
model = InstructBlipForConditionalGeneration.from_pretrained(model_id)
processor = InstructBlipProcessor.from_pretrained(model_id)

# Move the model to the specified device
model.to(device)

# Define the prompts
prompts = {
    "Recognised Food Item": "Which is the food item in the image?",
    "Recipe generated": "Generate a recipe for the dish shown in the image.",
    "List of ingredients predicted": "Provide the list of ingredients.",
    "Step by step instructions generated": "Provide step-by-step instructions."
}

# Function to process each prompt
def ask_instructblip(image_path, prompt):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

# Load the CSV file
csv_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/30_Dishes - newfilefiltered.csv'
data = pd.read_csv(csv_path)

# Save responses to the CSV file
output_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec_response/responses_instructblip.csv'

responses = []

for index, row in data.iterrows():
    image_name = row['Image_Name']
    image_path = f'/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/Images (finaL)/{image_name}.jpg'
    
    response_dict = {
        'Image_Name': image_name,
        'Recognised Food Item': '',
        'Recipe generated': '',
        'List of ingredients predicted': '',
        'Step by step instructions generated': ''
    }
    
    for key, prompt in prompts.items():
        print(f"Processing Image: {image_path} | Prompt: {prompt}")
        response = ask_instructblip(image_path, prompt)
        print(f"Response: {response}")
        response_dict[key] = response
    
    responses.append(response_dict)

# Convert responses to a DataFrame and save to CSV
response_df = pd.DataFrame(responses)
response_df.to_csv(output_path, index=False)

print("Processing complete. Responses saved to:", output_path)
