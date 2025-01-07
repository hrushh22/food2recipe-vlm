import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
from PIL import Image
import pandas as pd

# Set device to CUDA:1
device = torch.device("cuda:1")
torch.cuda.set_device(1)

# Load PaLI-Gemma model and processor
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Load and configure the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8, 
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":1})
model = get_peft_model(model, lora_config)

# Define the prompts
prompts = [
    "Which is the food item in the image?",
    "Generate a recipe for the dish shown in the image.",
    "Provide the list of ingredients.",
    "Provide step-by-step instructions."
]

# Load the CSV file
csv_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/30_Dishes - newfilefiltered.csv'
data = pd.read_csv(csv_path)

# Save responses to the CSV file
output_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec_response/PaliGemma_responses.csv'

# Function to run inference
def run_inference(image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5000, num_beams=5, early_stopping=True)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

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
        response = run_inference(image, prompt)
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
