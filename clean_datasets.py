import re

input_file = "datasets/NAME OF DATASET.txt"
output_file = f"{input_file}_cleaned.txt"

# Regular expression for URLs
url_pattern = re.compile(r'https?://\S+|www\.\S+')

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Replace URLs with empty string
cleaned_text = re.sub(url_pattern, '', text)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"Removed all links from '{input_file}' and saved to '{output_file}'")
