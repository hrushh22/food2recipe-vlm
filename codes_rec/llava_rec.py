# Model testing

from huggingface_hub import login
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline
import pandas as pd

# Login to Hugging Face Hub
login(token="your_hugging_face_token")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Set device to CUDA:0
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# Load the model and pipeline
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

def process_prompt(image_path, prompt):
    try:
        image = Image.open(image_path).convert('RGB')
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        outputs = pipe(image, prompt=formatted_prompt, generate_kwargs={"max_new_tokens": 5000})
        return outputs[0]["generated_text"]
    except Exception as e:
        return f"Error processing image: {e}"

# Load CSV file
csv_path = "/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/30_Dishes - newfilefiltered.csv"
df = pd.read_csv(csv_path)

# Initialize columns for responses
df['Recognised Food Item'] = ""
df['Recipe Generated'] = ""
df['List of Ingredients Predicted'] = ""
df['Step by Step Instructions Generated'] = ""

# Define prompts
prompts = [
    "Which is the food item in the image?",
    "Generate a recipe for the dish shown in the image.",
    "Provide the list of ingredients.",
    "Provide step-by-step instructions."
]

for index, row in df.iterrows():
    image_path = f"/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec/Images (finaL)/{row['Image_Name']}.jpg"
    print(f"Processing Image: {image_path}")
    try:
        responses = []
        for prompt in prompts:
            response = process_prompt(image_path, prompt)
            responses.append(response)
            print(f"Prompt: {prompt}\nResponse: {response}\n")

        # Assign responses to corresponding columns
        df.at[index, 'Recognised Food Item'] = responses[0]
        df.at[index, 'Recipe Generated'] = responses[1]
        df.at[index, 'List of Ingredients Predicted'] = responses[2]
        df.at[index, 'Step by Step Instructions Generated'] = responses[3]
    except Exception as e:
        print(f"Error processing image: {image_path}. Error: {e}")

# Save the updated dataframe to a new CSV file
output_csv_path = "/mnt/Data/punitsingh_1801/Hrushik/VLM/food_rec_data/FoodRec_response/responses_llava.csv"
df.to_csv(output_csv_path, index=False)

print("Processing complete. Responses saved to:", output_csv_path)
