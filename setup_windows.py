import sys
import importlib

def check_imports():
    """Sprawdza czy wszystkie wymagane biblioteki są zainstalowane"""
    required_modules = [
        'pygame', 'numpy', 'cv2', 'torch', 'matplotlib'
    ]
    
    print("Sprawdzanie zależności...")
    missing = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module} - OK")
        except ImportError:
            print(f"✗ {module} - BRAK")
            missing.append(module)
    
    if missing:
        print(f"\nBrakujące moduły: {', '.join(missing)}")
        print("Zainstaluj je używając: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ Wszystkie zależności są zainstalowane!")
        return True

def check_cuda():
    """Sprawdza dostępność CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA dostępne - GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("! CUDA niedostępne - będzie używany CPU")
            return False
    except:
        print("! Nie można sprawdzić CUDA")
        return False

if __name__ == "__main__":
    print("=== Sprawdzanie środowiska dla Tetris RL ===\n")
    
    if check_imports():
        check_cuda()
        print("\n=== Gotowe do uruchomienia! ===")
        
        # Test prostego importu
        try:
            from tetris_env import TetrisEnv
            from agent import ImprovedTetrisAgent
            print("✓ Moduły projektu importują się poprawnie")
        except Exception as e:
            print(f"✗ Problem z importem modułów projektu: {e}")
    else:
        print("\n=== Najpierw zainstaluj brakujące zależności ===")