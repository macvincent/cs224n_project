import json

file1_path = "input.txt" # or .nltktok
file2_path = "output.txt"
output_file_path = "output.jsonl"

with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_file_path, 'w') as output_file:
    for line1, line2 in zip(file1, file2):
        line1 = line1.strip()
        line2 = line2.strip()
        data = {"text":f"""<s>[INST] <<SYS>>\nYou are a model which converts all of my normal text into shakespearean styled text \n<</SYS>>\n\n {line1} [/INST] {line2} </s>"""}
        output_file.write(json.dumps(data) + '\n')