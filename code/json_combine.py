import json
import os

input_dir = "large_dataset_results"  # replace with your actual path
prefix = "mushroom_934"  # only process files starting with this

combined = []

for filename in os.listdir(input_dir):
    if filename.endswith(".json") and filename.lower().startswith(prefix.lower()):
        full_path = os.path.join(input_dir, filename)
        with open(full_path) as f:
            try:
                data = json.load(f)
                combined.append(data)
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

output_file = f"{prefix}_combined_results.json"
with open(output_file, "w") as f:
    json.dump(combined, f, indent=2)

print(f"✅ Combined {len(combined)} JSON files into {output_file}")
