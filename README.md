# 🧠 VLM-Food Recognition & Recipe Gen

> 📊 Zero-shot evaluation of 5 Vision-Language Models for food recognition and recipe generation using 13.5K annotated food images

## 🎯 Core Objectives
- `→` Zero-shot food recognition capabilities assessment
- `→` Recipe generation quality evaluation
- `→` Cross-model performance benchmarking
- `→` Domain-specific VLM behavior analysis

## 🔬 Technical Stack
```
VLMs        → LLaVA | LLAMA-3 | InstructBLIP | BLIP2 | PaLI-GEMMA
Dataset     → 13,582 images + paired recipes
Evaluation  → ROUGE-L + Human Expert Scoring
Methodology → Zero-shot prompting with sequential evaluation
```

## 📈 Performance Metrics

### ROUGE-L Scores
```
LLaVA       → 0.468 ⭐
LLAMA-3     → 0.352
InstructBLIP → 0.256
BLIP2       → 0.101
PaLI-GEMMA  → 0.182
```

### Expert Evaluation (n=10)
```
Metrics → {Ingredient_Accuracy, Instruction_Coherence, Cultural_Relevance, Recipe_Usability}
Scale   → [1-5]
Top     → LLaVA (μ=4.3/5.0)
```

## 🔋 Key Features
```python
# Zero-shot prompting pipeline
↳ Food Recognition  # Image → Food Item Identification
↳ Recipe Gen       # Image → Complete Recipe Generation
↳ Ingredient List  # Image → Ingredient Extraction
↳ Instructions     # Image → Step-by-Step Cooking Guide
```

## 🛠 Evaluation Framework
```
Input → Food Image
  ↓
Stage I   → Prompt Definition
Stage II  → Image Description
Stage III → Zero-Shot Response Generation
Stage IV  → Multi-Model Comparison
  ↓
Output → {Recognition_Score, Recipe_Quality, ROUGE-L}
```

## 📊 Dataset Structure
```
Food_Dataset/
  ├── 📸 images/      # 13.5K food images
  ├── 📝 recipes/     # Paired recipes
  ├── 🏷️ metadata/    # Image-recipe mappings
  └── 📑 annotations/ # Expert evaluations
```

## 🎯 Unique Contributions
- First comprehensive zero-shot VLM evaluation for culinary tasks
- Novel evaluation framework for recipe generation
- Cultural relevance assessment in AI-generated recipes
- Quantitative benchmarking of VLM food recognition
