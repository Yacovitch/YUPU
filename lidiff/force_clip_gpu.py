"""
This script modifies the CLIP model loading function to force GPU usage.
Run this script before training to patch the CLIP models.
"""
import os
import sys

def patch_clip_loader():
    # Find the CLIP model.py file
    try:
        # Try different potential locations for the CLIP module
        import lidiff.models.clip as clip
        clip_dir = os.path.dirname(clip.__file__)
        model_path = os.path.join(clip_dir, "model.py")
        
        if not os.path.exists(model_path):
            print(f"CLIP model.py not found at {model_path}")
            return False
        
        print(f"Found CLIP model.py at {model_path}")
        
        # Read the original file
        with open(model_path, 'r') as f:
            code = f.read()
        
        # Check if we need to patch
        if "# PATCHED FOR CUDA PRIORITY" in code:
            print("CLIP model.py already patched")
            return True
        
        # Find the load function
        load_func_start = code.find("def load(")
        if load_func_start == -1:
            print("Could not find 'def load(' in CLIP model.py")
            return False
        
        # Find the beginning of the function body
        func_body_start = code.find(":", load_func_start)
        if func_body_start == -1:
            print("Could not find function body start")
            return False
        
        # Insert our patch after the function signature
        patched_code = (
            code[:func_body_start+1] + 
            "\n    # PATCHED FOR CUDA PRIORITY\n" +
            "    if device == 'cpu' and torch.cuda.is_available():\n" +
            "        print('CLIP: Overriding CPU device with CUDA')\n" +
            "        device = 'cuda'\n" +
            code[func_body_start+1:]
        )
        
        # Write back the patched file
        with open(model_path, 'w') as f:
            f.write(patched_code)
        
        print("Successfully patched CLIP model.py to prioritize CUDA")
        return True
    
    except Exception as e:
        print(f"Error patching CLIP: {str(e)}")
        return False

if __name__ == "__main__":
    success = patch_clip_loader()
    sys.exit(0 if success else 1) 