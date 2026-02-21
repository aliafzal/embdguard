# check_env.py
import sys

def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        sys.exit(1)

check("torch",        lambda: __import__("torch"))
check("torchrec",     lambda: __import__("torchrec"))
check("fbgemm_gpu",   lambda: __import__("fbgemm_gpu"))
check("pandas",       lambda: __import__("pandas"))
check("numpy",        lambda: __import__("numpy"))
check("sklearn",      lambda: __import__("sklearn"))
check("tqdm",         lambda: __import__("tqdm"))

# Verify key TorchRec DMP imports
check("torchrec.distributed.model_parallel", lambda: __import__("torchrec.distributed.model_parallel"))
check("torchrec.optim.rowwise_adagrad",      lambda: __import__("torchrec.optim.rowwise_adagrad"))
check("torchrec.optim.keyed",                lambda: __import__("torchrec.optim.keyed"))

import torch
import torchrec
print(f"\n  torch:    {torch.__version__}")
print(f"  torchrec: {torchrec.__version__}")
print(f"  CUDA:     {torch.cuda.is_available()}")
print("\nAll dependencies OK.")
