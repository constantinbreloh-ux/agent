# Kaggle script with internet on, no GPU: builds every wheel vLLM needs, for offline install in gga-autopilot.
import subprocess
import sys

VLLM = 'vllm==0.34.0'
subprocess.run([sys.executable, '-m', 'pip', '--version'], check=True)
subprocess.run(f'{sys.executable} -m pip list 2>/dev/null | grep -iE "^(torch|transformers|compressed)"', shell=True)
subprocess.run([sys.executable, '-m', 'pip', 'wheel', '-q', VLLM, '-w', '/kaggle/working/wheels'], check=True)
subprocess.run('du -sh /kaggle/working/wheels && ls /kaggle/working/wheels | head -300', shell=True)
