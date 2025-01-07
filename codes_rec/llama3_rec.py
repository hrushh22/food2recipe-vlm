import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import pandas as pd

device = torch.device("cuda:1")
torch.cuda.set_device(1)

# Load model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

# Load the CSV file
csv_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/30_Dishes - newfilefiltered.csv'
data = pd.read_csv(csv_path)

# Define the prompts
prompts = [
    "Which is the food item in the image?",
    "Generate a recipe for the dish shown in the image.",
    "Provide the list of ingredients.",
    "Provide step-by-step instructions."
]

# Function to process each prompt
def process_prompt(prompt, image):
    msgs = [{'role': 'user', 'content': prompt}]
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
    )
    return res

# Save responses to the CSV file
output_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec_response/llama3_responses.csv'

responses = []

for index, row in data.iterrows():
    image_name = row['Image_Name']
    image_path = f'/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/Images (finaL)/{image_name}.jpg'
    
    image = Image.open(image_path).convert('RGB')
    
    response_dict = {
        'Image_Name': image_name,
        'Recognised Food Item': '',
        'Recipe generated': '',
        'List of ingredients predicted': '',
        'Step by step instructions generated': ''
    }
    
    for i, prompt in enumerate(prompts):
        print(f"Processing Image: {image_path} | Prompt: {prompt}")
        response = process_prompt(prompt, image)
        print(f"Response: {response}")
        if i == 0:
            response_dict['Recognised Food Item'] = response
        elif i == 1:
            response_dict['Recipe generated'] = response
        elif i == 2:
            response_dict['List of ingredients predicted'] = response
        elif i == 3:
            response_dict['Step by step instructions generated'] = response
    
    responses.append(response_dict)

# Convert responses to a DataFrame and save to CSV
response_df = pd.DataFrame(responses)
response_df.to_csv(output_path, index=False)

print("Processing complete. Responses saved to:", output_path)