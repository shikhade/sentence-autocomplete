This project fine-tunes a pre-trained GPT-2 model on the WikiText-2 dataset to build a simple Next-Word Predictor.

# Project Summary
-	Model: “GPT-2” from Hugging Face Transformers
-	Dataset: 'WikiText-2'
-	Training Epochs: 3
-	Final Loss: 2.61
-	Training Time: ~285 minutes (on laptop CPU)
-	Inference Enabled: yes
> Due to hardware constraints, training was limited to 3 epochs. Full training over 25–30 epochs is expected to improve performance significantly.

# Results
Epoch	Training Loss
 1	       3.05
 2	       2.80
 3	       2.61
 10	      ~1.8 (expected)
 20	      ~1.4 (expected)


## 🔁 Inference Demo
~python code
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2-wikitext2-final")
tokenizer = AutoTokenizer.from_pretrained("gpt2-wikitext2-final")

input = tokenizer("The quick brown", return_tensors="pt")
output = model.generate(**input, max_length=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
