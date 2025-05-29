"""
test_all.py - Test all modules after fixes
Run this to verify all modules work correctly
"""

import subprocess
import sys
import os

def test_module(module_name):
    """Test a single module"""
    print(f"\n{'='*60}")
    print(f"Testing {module_name}.py")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, f"{module_name}.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"✓ {module_name} passed")
            return True
        else:
            print(f"✗ {module_name} failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {module_name} timed out")
        return False
    except Exception as e:
        print(f"✗ {module_name} error: {e}")
        return False

def main():
    """Test all modules in correct order"""
    modules = [
        "core",
        "perturbations", 
        "collapse",
        "model",
        "data",
        "inference",
        "trainer"
    ]
    
    print("Field-Theoretic Language Model - Module Tests")
    print("=" * 60)
    
    results = {}
    for module in modules:
        if os.path.exists(f"{module}.py"):
            results[module] = test_module(module)
        else:
            print(f"Warning: {module}.py not found")
            results[module] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for module, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{module:15} {status}")
    
    print(f"\nTotal: {passed}/{total} modules passed")
    
    if passed == total:
        print("\n🎉 All modules validated successfully!")
        print("\nYou can now run:")
        print("  python main.py train --dataset wikitext-2 --num-steps 1000")
        print("  python main.py generate 'The quantum field'")
    else:
        print("\n⚠️  Some modules failed. Check the errors above.")

if __name__ == "__main__":
    main()