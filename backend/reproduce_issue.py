
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

try:
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
except Exception as e:
    print(f"Error checks cuda: {e}")

print("Checking torch.cpu...")
try:
    print(f"hasattr(torch.cpu, 'is_bf16_supported'): {hasattr(torch.cpu, 'is_bf16_supported')}")
except Exception as e:
    print(f"Error checking hasattr: {e}")

try:
    if hasattr(torch.cpu, 'is_bf16_supported'):
        print(f"torch.cpu.is_bf16_supported(): {torch.cpu.is_bf16_supported()}")
    else:
        print("torch.cpu.is_bf16_supported not present")
except Exception as e:
    print(f"Error calling is_bf16_supported: {e}")

try:
    # Mimic the code in llm_service.py
    use_bf16 = False
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()
    elif hasattr(torch.cpu, 'is_bf16_supported'):
        use_bf16 = torch.cpu.is_bf16_supported()
    print(f"Resulting use_bf16: {use_bf16}")
except Exception as e:
    print(f"Error in logic block: {e}")
