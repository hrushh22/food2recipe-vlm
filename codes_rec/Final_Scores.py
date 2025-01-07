# import pandas as pd
# from collections import Counter
# from math import exp, log
# import chardet

# def simple_bleu(reference, hypothesis, max_n=4):
#     """
#     Calculates a simple BLEU score for a single sentence.
#     """
#     ref_counts = Counter(reference)
#     hyp_counts = Counter(hypothesis)
    
#     clipped_counts = {word: min(count, ref_counts[word]) for word, count in hyp_counts.items()}
    
#     total_matches = sum(clipped_counts.values())
#     total_words = len(hypothesis)
    
#     if total_words == 0:
#         return 0
    
#     brevity_penalty = min(1, exp(1 - len(reference) / max(len(hypothesis), 1)))
    
#     score = brevity_penalty * exp(log(max(total_matches, 1) / max(total_words, 1)))
    
#     return score

# def simple_rouge_l(reference, hypothesis):
#     """
#     Calculates a simple ROUGE-L score for a single sentence.
#     """
#     def lcs(X, Y):
#         m = len(X)
#         n = len(Y)
#         L = [[0] * (n + 1) for _ in range(m + 1)]
#         for i in range(1, m + 1):
#             for j in range(1, n + 1):
#                 if X[i-1] == Y[j-1]:
#                     L[i][j] = L[i-1][j-1] + 1
#                 else:
#                     L[i][j] = max(L[i-1][j], L[i][j-1])
#         return L[m][n]

#     lcs_length = lcs(reference, hypothesis)
#     if len(reference) == 0 or len(hypothesis) == 0:
#         return 0
#     return lcs_length / max(len(reference), len(hypothesis))

# # Load the CSV file with encoding detection
# file_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/dataset/Food_eval.csv'

# # Detect the file encoding
# with open(file_path, 'rb') as file:
#     raw_data = file.read()

# detected = chardet.detect(raw_data)
# encoding = detected['encoding']

# # Load the CSV file with the detected encoding
# df = pd.read_csv(file_path, encoding=encoding)

# # Clean up the DataFrame
# df = df.dropna(how='all')  # Remove completely empty rows
# df['Models'] = df['Models'].str.strip()  # Remove trailing newlines

# # Models list (make sure these match exactly with the cleaned model names in the CSV)
# models = [
#     'MiniCPM-Llama3-V-2_5-int4',
#     'llava-1.5-7b-hf',
#     'instructblip-flan-t5-xl',
#     'paligemma-3b-pt-224',
#     'blip2-opt-2.7b'
# ]

# # Prepare an empty DataFrame for results
# results = []

# # Loop through each model
# for model in models:
#     print(f"Processing model: {model}")
#     model_data = df[df['Models'] == model]
#     print(f"Number of rows for {model}: {len(model_data)}")
    
#     if len(model_data) == 0:
#         print(f"No data found for model {model}. Skipping...")
#         continue

#     bleu_scores = []
#     rouge_scores = []

#     # Loop through the available examples
#     for i in range(min(6, len(model_data))):
#         # Get the example number as a string
#         example_num = str(i+1)

#         # Extract the text for comparison
#         recognised_food_item = str(model_data['Recognised_Food_Item'].iloc[i])
#         food_item = str(model_data['Food_Item'].iloc[i])
#         predicted_ingredients = str(model_data['Predicted_Ingredients'].iloc[i])
#         ingredients = str(model_data['Ingredients'].iloc[i])
#         recipe_generated = str(model_data['Recipe_Generated'].iloc[i])
#         recipe = str(model_data['Recipe'].iloc[i])

#         # Calculate BLEU and ROUGE scores for each pair
#         bleu_recognised_vs_food = simple_bleu(food_item.split(), recognised_food_item.split())
#         rouge_recognised_vs_food = simple_rouge_l(food_item.split(), recognised_food_item.split())

#         bleu_predicted_vs_ingredients = simple_bleu(ingredients.split(), predicted_ingredients.split())
#         rouge_predicted_vs_ingredients = simple_rouge_l(ingredients.split(), predicted_ingredients.split())

#         bleu_recipe_vs_generated = simple_bleu(recipe.split(), recipe_generated.split())
#         rouge_recipe_vs_generated = simple_rouge_l(recipe.split(), recipe_generated.split())

#         # Append scores to list
#         bleu_scores.extend([bleu_recognised_vs_food, bleu_predicted_vs_ingredients, bleu_recipe_vs_generated])
#         rouge_scores.extend([rouge_recognised_vs_food, rouge_predicted_vs_ingredients, rouge_recipe_vs_generated])

#         # Save individual results
#         results.append({
#             'Model': model,
#             'Example': example_num,
#             'BLEU_Recognised_vs_Food': bleu_recognised_vs_food,
#             'ROUGE_Recognised_vs_Food': rouge_recognised_vs_food,
#             'BLEU_Predicted_vs_Ingredients': bleu_predicted_vs_ingredients,
#             'ROUGE_Predicted_vs_Ingredients': rouge_predicted_vs_ingredients,
#             'BLEU_Recipe_vs_Generated': bleu_recipe_vs_generated,
#             'ROUGE_Recipe_vs_Generated': rouge_recipe_vs_generated
#         })

#     # Calculate the average scores for the model
#     avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
#     avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

#     # Append the overall averages
#     results.append({
#         'Model': model,
#         'Example': 'Overall Average',
#         'BLEU_Recognised_vs_Food': avg_bleu,
#         'ROUGE_Recognised_vs_Food': avg_rouge,
#         'BLEU_Predicted_vs_Ingredients': avg_bleu,
#         'ROUGE_Predicted_vs_Ingredients': avg_rouge,
#         'BLEU_Recipe_vs_Generated': avg_bleu,
#         'ROUGE_Recipe_vs_Generated': avg_rouge
#     })

# # Convert results to DataFrame
# results_df = pd.DataFrame(results)

# # Save to CSV
# results_df.to_csv('/mnt/Data/punitsingh_1801/Hrushik/VLM/FoodRec/FoodRecD/codes_rec/Food_Eval_Results.csv', index=False)

# print("Evaluation complete and saved to 'Food_Eval_Results.csv'")







# Final method to calculate 

import pandas as pd
from evaluate import load
from IPython.display import display
import chardet

# Load the BLEU and ROUGE metrics
bleu = load('bleu')
rouge = load('rouge')

# Load the CSV file with encoding detection
file_path = '/mnt/Data/punitsingh_1801/Hrushik/VLM/dataset/Food_eval.csv'

# Detect the file encoding
with open(file_path, 'rb') as file:
    raw_data = file.read()

detected = chardet.detect(raw_data)
encoding = detected['encoding']

# Load the CSV file with the detected encoding
df = pd.read_csv(file_path, encoding=encoding)

# Clean up the DataFrame
df = df.dropna(how='all')  # Remove completely empty rows
df['Models'] = df['Models'].str.strip()  # Remove trailing newlines

# Models list (make sure these match exactly with the cleaned model names in the CSV)
models = [
    'MiniCPM-Llama3-V-2_5-int4',
    'llava-1.5-7b-hf',
    'instructblip-flan-t5-xl',
    'paligemma-3b-pt-224',
    'blip2-opt-2.7b'
]

# Prepare an empty DataFrame for results
results = []

# Loop through each model
for model in models:
    print(f"Processing model: {model}")
    model_data = df[df['Models'] == model]
    print(f"Number of rows for {model}: {len(model_data)}")
    
    if len(model_data) == 0:
        print(f"No data found for model {model}. Skipping...")
        continue

    bleu_scores = []
    rouge_scores = []

    # Loop through the available examples
    for i in range(min(6, len(model_data))):
        # Get the example number as a string
        example_num = str(i+1)

        # Extract the text for comparison
        recognised_food_item = str(model_data['Recognised_Food_Item'].iloc[i])
        food_item = str(model_data['Food_Item'].iloc[i])
        predicted_ingredients = str(model_data['Predicted_Ingredients'].iloc[i])
        ingredients = str(model_data['Ingredients'].iloc[i])
        recipe_generated = str(model_data['Recipe_Generated'].iloc[i])
        recipe = str(model_data['Recipe'].iloc[i])

        # Debugging: Print input for rouge
        print(f"\nDebugging Inputs for ROUGE (Example {i+1}):")
        print(f"Recognised Food Item: {recognised_food_item}")
        print(f"Food Item: {food_item}")

        # Calculate BLEU and ROUGE scores for each pair
        bleu_recognised_vs_food = bleu.compute(predictions=[recognised_food_item], references=[food_item])['bleu']
        rouge_result = rouge.compute(predictions=[recognised_food_item], references=[food_item])

        # Directly access the score from rouge_result
        rouge_recognised_vs_food = rouge_result['rougeL']

        # Debugging: Print the rouge_result to understand its structure
        print(f"ROUGE result: {rouge_result}")

        bleu_predicted_vs_ingredients = bleu.compute(predictions=[predicted_ingredients], references=[ingredients])['bleu']
        rouge_predicted_vs_ingredients = rouge.compute(predictions=[predicted_ingredients], references=[ingredients])['rougeL']

        bleu_recipe_vs_generated = bleu.compute(predictions=[recipe_generated], references=[recipe])['bleu']
        rouge_recipe_vs_generated = rouge.compute(predictions=[recipe_generated], references=[recipe])['rougeL']

        # Append scores to list
        bleu_scores.extend([bleu_recognised_vs_food, bleu_predicted_vs_ingredients, bleu_recipe_vs_generated])
        rouge_scores.extend([rouge_recognised_vs_food, rouge_predicted_vs_ingredients, rouge_recipe_vs_generated])

        # Save individual results
        results.append({
            'Model': model,
            'Example': example_num,
            'BLEU_Recognised_vs_Food': bleu_recognised_vs_food,
            'ROUGE_Recognised_vs_Food': rouge_recognised_vs_food,
            'BLEU_Predicted_vs_Ingredients': bleu_predicted_vs_ingredients,
            'ROUGE_Predicted_vs_Ingredients': rouge_predicted_vs_ingredients,
            'BLEU_Recipe_vs_Generated': bleu_recipe_vs_generated,
            'ROUGE_Recipe_vs_Generated': rouge_recipe_vs_generated
        })

    # Calculate the average scores for the model
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

    # Append the overall averages
    results.append({
        'Model': model,
        'Example': 'Overall Average',
        'BLEU_Recognised_vs_Food': avg_bleu,
        'ROUGE_Recognised_vs_Food': avg_rouge,
        'BLEU_Predicted_vs_Ingredients': avg_bleu,
        'ROUGE_Predicted_vs_Ingredients': avg_rouge,
        'BLEU_Recipe_vs_Generated': avg_bleu,
        'ROUGE_Recipe_vs_Generated': avg_rouge
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('/mnt/Data/punitsingh_1801/Hrushik/VLM/FoodRec/FoodRecD/codes_rec/Final_Food_Eval_Results.csv', index=False)

print("Evaluation complete and saved to 'Food_Eval_Results.csv'")
