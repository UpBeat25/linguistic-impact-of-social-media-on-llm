import os
import random

# ========== CONFIG ==========
conversation_files = [
    "datasets/conversation/human_chat.txt",
    "datasets/conversation/reddit_casual_conversation.txt"
]

singleline_files = [
    "datasets/discord_chat_messages_only_cleaned.txt",
    "datasets/gametox_cleaned.txt",
    "datasets/suspicious_communication_on_social_platforms_cleaned.txt"
]

output_dir = "datasets/stages"
num_stages = 5  # number of stage files

# Create subfolders
os.makedirs(f"{output_dir}/conversation", exist_ok=True)
os.makedirs(f"{output_dir}/singleline", exist_ok=True)

# ========== LOAD CONVERSATION PAIRS ==========
all_conv = []
for file in conversation_files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        pairs = [p.strip() for p in text.split("{---}") if p.strip()]
        all_conv.extend(pairs)

random.shuffle(all_conv)

# ========== LOAD SINGLE-LINE TEXTS ==========
all_single = []
for file in singleline_files:
    with open(file, "r", encoding="utf-8") as f:
        all_single.extend([line.strip() for line in f.readlines() if line.strip()])

random.shuffle(all_single)

# ========== CALCULATIONS ==========
total_conv = len(all_conv)

# Conversation lines per stage (equal distribution)
conv_per_stage = total_conv // num_stages  

# Single-line limit (20% of stage size)
single_per_stage = conv_per_stage // 4

# Trim excess single-lines
all_single = all_single[: single_per_stage * num_stages]

# ========== GENERATE STAGE FILES ==========
for i in range(num_stages):

    # Conversation slice
    conv_start = i * conv_per_stage
    conv_end = (i + 1) * conv_per_stage
    conv_chunk = all_conv[conv_start:conv_end]

    # Single-line slice
    single_start = i * single_per_stage
    single_end = (i + 1) * single_per_stage
    single_chunk = all_single[single_start:single_end]

    # Output paths
    conv_path = os.path.join(output_dir, f"conversation/stage_{i+1}_conversation.txt")
    single_path = os.path.join(output_dir, f"singleline/stage_{i+1}_singleline.txt")

    # Save conversations
    with open(conv_path, "w", encoding="utf-8") as f:
        f.write("\n{---}\n".join(conv_chunk))

    # Save single lines
    with open(single_path, "w", encoding="utf-8") as f:
        f.write("\n".join(single_chunk))

print("✅ All stage files created successfully!")
print(f"Conversation per stage: {conv_per_stage}")
print(f"Single-lines per stage: {single_per_stage}")
