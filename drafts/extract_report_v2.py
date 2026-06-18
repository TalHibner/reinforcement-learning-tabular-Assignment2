import json
import base64

with open('HW2_2026B_KeyDoorLava_final.ipynb', 'r') as f:
    nb = json.load(f)

md_lines = [
    "# Reinforcement Learning: Mid Semester Project - 2026 B",
    "**Gilad Ticher 318770039 & Tal Hibner 026548446**",
    "\n---\n"
]

img_idx = 0
for cell in nb['cells']:
    
    # 1. Extract Markdown sections (Environments + Discussion + Best Settings)
    if cell['cell_type'] == 'markdown':
        src = "".join(cell.get('source', []))
        # Add 'Environments' sections
        if "Environments" in src or "Environment 1:" in src or "Environment 2:" in src:
            if "Display utils" not in src: # avoid matching other random headers
                md_lines.append(src + "\n")
        # Add Discussion and Best Settings
        elif "8. Discussion" in src or "8.1" in src or "8.2" in src or "8.3" in src or "8.4" in src or "8.5" in src or "9. Best Settings" in src:
            md_lines.append(src + "\n")

    # 2. Extract Greedy Evaluation tables AND Training Plots
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if 'outputs' in cell:
            for out in cell['outputs']:
                # Extract Greedy Eval Text
                if 'text' in out:
                    text_content = "".join(out['text'])
                    if "Greedy Evaluation" in text_content:
                        md_lines.append("```text\n" + text_content.strip() + "\n```\n")
                
                # Extract ONLY Training Plots (exclude initial state env.render() images)
                if 'data' in out and 'image/png' in out['data']:
                    # We only extract images from cells that explicitly plot results
                    if "plot_training_results" in src or "plt.show" in src:
                        img_data = out['data']['image/png']
                        img_filename = f'final_report_img_clean_{img_idx}.png'
                        with open(img_filename, 'wb') as img_f:
                            img_f.write(base64.b64decode(img_data))
                        
                        # Add a clean title based on the environment
                        if "EmptyEnv" in src:
                            md_lines.append(f"**EmptyEnv Training Progress**\n![EmptyEnv Plot]({img_filename})\n")
                        else:
                            md_lines.append(f"**KeyDoorLavaEnv Training Progress**\n![KeyDoorLavaEnv Plot]({img_filename})\n")
                        img_idx += 1

with open('report_026548446_318770039.md', 'w') as f:
    f.write("\n".join(md_lines))

print("Extraction complete.")
