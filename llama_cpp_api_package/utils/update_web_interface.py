import os
import re
from pathlib import Path

def get_model_files():
    """Get all GGUF model files in the models directory"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("Models directory not found")
        return []
    
    return sorted([f for f in models_dir.glob("*.gguf") if f.is_file()])

def update_index_html():
    """Update the index.html file with model options based on available models"""
    index_path = Path("static/index.html")
    if not index_path.exists():
        print("static/index.html not found")
        return False
    
    model_files = get_model_files()
    if not model_files:
        print("No GGUF model files found in models directory")
        return False
    
    with open(index_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the select element for model paths
    select_pattern = r'<select class="form-select" id="modelPath" onchange="updateModelId\(\)">(.*?)</select>'
    select_match = re.search(select_pattern, content, re.DOTALL)
    
    if not select_match:
        print("Could not find model select element in index.html")
        return False
    
    # Create new options for each model
    options = []
    for model_file in model_files:
        model_name = model_file.stem
        model_id = model_name.split('.')[0]  # Use first part before dot as ID
        
        # Skip NSFW models
        if model_name.lower().startswith("nsfw"):
            continue
            
        options.append(f'<option value="models/{model_file.name}" data-id="{model_id}">{model_id}</option>')
    
    # If no safe models were found
    if not options:
        print("No safe models found (skipping NSFW models)")
        return False
    
    # Create the new select element content
    new_select = f'<select class="form-select" id="modelPath" onchange="updateModelId()">\n'
    new_select += '\n'.join(options)
    new_select += '\n</select>'
    
    # Replace the select element in the HTML
    new_content = re.sub(select_pattern, new_select, content, flags=re.DOTALL)
    
    # Save the updated HTML
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated index.html with {len(options)} model options")
    print("Models added:")
    for option in options:
        print(f"  - {option.split('data-id=\"')[1].split('\"')[0]}")
    
    return True

if __name__ == "__main__":
    print("Scanning for models...")
    model_files = get_model_files()
    
    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({file_size_mb:.1f} MB)")
    
    print("\nUpdating web interface...")
    if update_index_html():
        print("\nSuccessfully updated the web interface!")
        print("Start the server with: python llama_cpp_python.py")
    else:
        print("\nFailed to update the web interface.") 