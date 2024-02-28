import together
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
new_model_name = 'ayaan04@stanford.edu/llama-2-7b-chat-2024-02-28-01-18-37'
together.Models.start(new_model_name)
together.Models.ready(new_model_name)
output_file_path = "/Users/ayaan/Downloads/Archive (2)/output.txt"
with open("/Users/ayaan/Downloads/Archive (2)/test_original.nltktok", "r") as input_file:
    lines = input_file.readlines()
processed_lines = []
i = 0
for line in lines:
    test_chat_prompt = f"<s>[INST] <<SYS>> You are a model which converts all of my normal text into shakespearean styled text <</SYS>> {line} [/INST]"
    output = together.Complete.create(
        prompt = test_chat_prompt,
        model = new_model_name,
        max_tokens = 256,
        temperature = 0.6,
        top_k = 90,
        top_p = 0.8,
        repetition_penalty = 1.1,
        stop = ['</s>']
    )
    processed_lines.append(output['output']['choices'][0]['text'])
    time.sleep(1) # adjust this so we do not get network error. Also, EDIT CODE TO WRITE EVERY 200 ITERATIONS. IMP!!!
    print(i)
    i += 1

# print generated text
with open(output_file_path, "w") as output_file:
    output_file.write("\n".join(processed_lines))
    print(f"Output saved to '{output_file_path}'.")