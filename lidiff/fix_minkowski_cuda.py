"""
Script to manually rebuild MinkowskiEngine with CUDA support
"""
import os
import sys
import subprocess
import torch

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
        # Uninstall existing MinkowskiEngine
        print("Uninstalling existing MinkowskiEngine...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "MinkowskiEngine"])
        
        # Install MinkowskiEngine with CUDA support
        print("Installing MinkowskiEngine with CUDA support...")
        cmd = [
            sys.executable, "-m", "pip", "install", "-v", 
            "git+https://github.com/NVIDIA/MinkowskiEngine", 
            "--no-deps"
        ]
        
        subprocess.check_call(cmd, env=env)
        
        # Verify installation
        print("Verifying MinkowskiEngine CUDA support...")
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
            
    except Exception as e:
        print(f"Error rebuilding MinkowskiEngine: {str(e)}")
        return False

if __name__ == "__main__":
    success = install_minkowski_with_cuda()
    sys.exit(0 if success else 1) 