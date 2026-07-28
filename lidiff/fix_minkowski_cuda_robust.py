"""
Script to manually rebuild MinkowskiEngine with CUDA support - with robust error handling
"""
import os
import sys
import subprocess
import torch
import shutil
import site
import glob

def run_command(cmd, env=None, ignore_errors=False):
    """Run a command and return output"""
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            env=env, 
            check=not ignore_errors,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Command output: {result.stdout}")
        if result.stderr:
            print(f"Command error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return False

def force_remove_minkowski():
    """Force removal of MinkowskiEngine by finding and deleting its files"""
    try:
        # Try different approaches to find MinkowskiEngine files
        site_packages = site.getsitepackages()
        paths_to_check = site_packages + [os.path.expanduser("~/.local/lib/python3.8/site-packages")]
        
        removed_files = []
        
        # Find all MinkowskiEngine directories
        for site_dir in paths_to_check:
            if os.path.exists(site_dir):
                print(f"Checking site directory: {site_dir}")
                minkowski_paths = glob.glob(f"{site_dir}/MinkowskiEngine*")
                for path in minkowski_paths:
                    print(f"Found MinkowskiEngine at: {path}")
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    removed_files.append(path)
        
        # Find MinkowskiEngine in easy-install.pth
        for site_dir in paths_to_check:
            easy_install_path = os.path.join(site_dir, "easy-install.pth")
            if os.path.exists(easy_install_path):
                print(f"Modifying easy-install.pth: {easy_install_path}")
                with open(easy_install_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = [line for line in lines if "MinkowskiEngine" not in line]
                if len(new_lines) != len(lines):
                    with open(easy_install_path, 'w') as f:
                        f.writelines(new_lines)
                    print(f"Removed MinkowskiEngine entries from easy-install.pth")
        
        if removed_files:
            print(f"Successfully removed MinkowskiEngine files: {removed_files}")
            return True
        else:
            print("No MinkowskiEngine files found to remove")
            return False
    
    except Exception as e:
        print(f"Error during force removal: {str(e)}")
        return False

def install_minkowski_with_cuda():
    """
    Attempt to install MinkowskiEngine with CUDA support within the container.
    """
    print("=" * 50)
    print("REBUILDING MINKOWSKI ENGINE WITH CUDA SUPPORT")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot build CUDA version.")
        return False
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Set environment variables
    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = "8.9"  # For L40 GPUs
    
    print("Environment variables:")
    print(f"FORCE_CUDA: {env.get('FORCE_CUDA')}")
    print(f"TORCH_CUDA_ARCH_LIST: {env.get('TORCH_CUDA_ARCH_LIST')}")
    print(f"CUDA_HOME: {env.get('CUDA_HOME')}")
    
    try:
        # Try standard uninstall first
        print("Trying standard uninstall of MinkowskiEngine...")
        run_command([sys.executable, "-m", "pip", "uninstall", "-y", "MinkowskiEngine"], ignore_errors=True)
        
        # Force remove if standard uninstall fails
        print("Forcing removal of MinkowskiEngine files...")
        force_remove_minkowski()
        
        # Make sure pip is updated
        print("Updating pip...")
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], ignore_errors=True)
        
        # Install build dependencies
        print("Installing build dependencies...")
        run_command([
            sys.executable, "-m", "pip", "install", 
            "setuptools", "wheel", "ninja", "cmake"
        ], ignore_errors=True)
        
        # Install MinkowskiEngine with CUDA support
        print("Installing MinkowskiEngine with CUDA support...")
        success = run_command([
            sys.executable, "-m", "pip", "install", "-v", 
            "git+https://github.com/NVIDIA/MinkowskiEngine.git", 
            "--no-deps"
        ], env=env)
        
        if not success:
            print("First attempt failed, trying alternate repository...")
            # Try an alternate repository if the first fails
            success = run_command([
                sys.executable, "-m", "pip", "install", "-v", 
                "git+https://github.com/NVIDIA/MinkowskiEngine", 
                "--no-deps"
            ], env=env)
        
        # Verify installation
        print("Verifying MinkowskiEngine CUDA support...")
        try:
            import MinkowskiEngine as ME
            print(f"MinkowskiEngine version: {ME.__version__}")
            print(f"MinkowskiEngine CUDA enabled: {ME.is_cuda_available()}")
            
            if ME.is_cuda_available():
                print("SUCCESS: MinkowskiEngine now has CUDA support!")
                # Test creating a tensor on CUDA
                try:
                    coords = torch.IntTensor([[0, 0, 0], [0, 0, 1]])
                    feats = torch.FloatTensor([[1.0], [2.0]])
                    tensor = ME.SparseTensor(feats, coords, device=torch.device('cuda'))
                    print("Created CUDA SparseTensor successfully!")
                    return True
                except Exception as e:
                    print(f"Failed to create CUDA tensor: {str(e)}")
                    return False
            else:
                print("FAILURE: MinkowskiEngine still doesn't have CUDA support")
                return False
        except ImportError:
            print("Failed to import MinkowskiEngine after installation")
            return False
            
    except Exception as e:
        print(f"Error rebuilding MinkowskiEngine: {str(e)}")
        return False

if __name__ == "__main__":
    success = install_minkowski_with_cuda()
    sys.exit(0 if success else 1) 