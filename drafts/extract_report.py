import json
import base64
import os

with open('HW2_2026B_KeyDoorLava_final.ipynb', 'r') as f:
    nb = json.load(f)

md_lines = [
    "# Reinforcement Learning: Mid Semester Project - 2026 B",
    "**Gilad Ticher 318770039 & Tal Hibner 026548446**",
    "\n---\n"
]

img_idx = 0
for i, cell in enumerate(nb['cells']):
    
    # 1. Extract Greedy Evaluation text outputs
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for out in cell['outputs']:
            if 'text' in out:
                text_content = "".join(out['text'])
                if "Greedy Evaluation" in text_content:
                    md_lines.append("```text\n" + text_content.strip() + "\n```\n")
            
            # 2. Extract Training Graphs (PNGs)
            if 'data' in out and 'image/png' in out['data']:
                img_data = out['data']['image/png']
                
                # Try to avoid video frames if they exist as png
                # Usually training plots are large base64 strings
                if len(img_data) > 5000:
                    img_filename = f'final_report_img_{img_idx}.png'
                    with open(img_filename, 'wb') as img_f:
                        img_f.write(base64.b64decode(img_data))
                    
                    md_lines.append(f"![Result Graph]({img_filename})\n")
                    img_idx += 1

    # 3. Extract Section 8 and 9 Markdown
    if cell['cell_type'] == 'markdown':
        src = "".join(cell.get('source', []))
        if "8. Discussion" in src or "8.1" in src or "8.2" in src or "8.3" in src or "8.4" in src or "8.5" in src or "9. Best Settings" in src:
            md_lines.append(src + "\n")

with open('report_026548446_318770039.md', 'w') as f:
    f.write("\n".join(md_lines))

print("Extraction complete.")
