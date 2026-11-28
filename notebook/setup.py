# Setup: installs vLLM and Chromium offline and starts Gemma 4 31B (4-bit) in a vLLM server on localhost:8000.
import io, json, os, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

began = time.time()
root = Path('/kaggle/input')
harness = next(root.rglob('gga/protocol.py')).parents[1]
weights = next(root.rglob('gemma-4-31b-it-qat-w4a16-ct/*/model.safetensors')).parent
kit = next(root.rglob('kit/wheels')).parent
wheels = next(root.rglob('vllm-0.34.0-*.whl')).parent   # output of the notebook gga-vllm-wheels


def log(text):
    print(f'{time.time() - began:5.0f} s  {text}', flush=True)


# Chromium installs meanwhile. The input folder drops executable bits, so it is copied out first.
chromium = subprocess.Popen(f'cp -r {kit}/browsers /tmp && chmod -R +x /tmp/browsers'
                            f' && dpkg -i {kit}/debs/*.deb > /dev/null 2>&1', shell=True)
# The input folder is slow per stream but scales: read the weights in 16 streams, so vLLM loads from memory.
fd = os.open(weights / 'model.safetensors', os.O_RDONLY)
reading = ThreadPoolExecutor(16).map(lambda offset: os.pread(fd, 64 * 2**20, offset),
                                     range(0, os.fstat(fd).st_size, 64 * 2**20))
subprocess.run(f'pip install -q --no-index --find-links {wheels} --find-links {kit}/wheels vllm playwright',
               shell=True, check=True)
log('installed vLLM and Playwright')
subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1920x1080x24'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.path.insert(0, str(harness))
sys.path.insert(0, '/kaggle/working')   # for autopilot.py, written by the next cell
os.environ.update(GGA_ROOT=str(harness), GGA_SEED='0', DISPLAY=':99', PLAYWRIGHT_BROWSERS_PATH='/tmp/browsers')
list(reading)   # raises if a read failed
os.close(fd)
log('read the weights')

server = subprocess.Popen(
    ['vllm', 'serve', str(weights), '--port', '8000', '--served-model-name', 'gemma', '--max-model-len', '16384',
     '--gpu-memory-utilization', '0.9', '--max-num-seqs', '8', '--limit-mm-per-prompt', '{"image": 1}'],
    stdout=open('/kaggle/working/vllm.log', 'w'), stderr=subprocess.STDOUT)
while True:   # the competition allows 15 minutes for everything before the inference server listens
    try:
        urllib.request.urlopen('http://localhost:8000/health', timeout=2)
        break
    except OSError:
        if server.poll() is not None:
            sys.exit('vLLM exited:\n' + Path('/kaggle/working/vllm.log').read_text()[-3000:])
        time.sleep(2)
log('vLLM is serving Gemma')
if chromium.wait():
    raise RuntimeError('Installing Chromium failed')
log('ready')
