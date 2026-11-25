# Setup, from the starter notebook: installs Chromium and loads Gemma 4 31B (4-bit) in transformers.
import contextlib, io, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

PIECE = 64 * 2**20

root = Path('/kaggle/input')
harness = next(root.rglob('gga/protocol.py')).parents[1]
weights = next(root.rglob('gemma-4-31b-it-qat-w4a16-ct/*/model.safetensors')).parent
kit = next(root.rglob('kit/wheels')).parent


def tick(bar):
    while not bar.disable:   # redraws the bar every second, so its clock runs during steps that block
        bar.refresh()
        time.sleep(1)


with tqdm(total=100, desc='Installing packages', file=sys.stdout,
          bar_format='{desc:<20} {percentage:3.0f}%|{bar:30}| {elapsed}') as bar:
    threading.Thread(target=tick, args=[bar], daemon=True).start()
    # Chromium installs while Gemma loads. The input folder drops executable bits, so it is copied out first.
    chromium = subprocess.Popen(f'cp -r {kit}/browsers /tmp && chmod -R +x /tmp/browsers'
                                f' && dpkg -i {kit}/debs/*.deb > /dev/null 2>&1', shell=True)
    subprocess.run(f'pip install -q --no-index --find-links {kit}/wheels playwright compressed-tensors',
                   shell=True, check=True)
    subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1920x1080x24'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.path.insert(0, str(harness))
    sys.path.insert(0, '/kaggle/working')   # for autopilot.py, written by the next cell
    os.environ.update(GGA_ROOT=str(harness), GGA_SEED='0', DISPLAY=':99', PLAYWRIGHT_BROWSERS_PATH='/tmp/browsers')
    bar.update(5)

    # The input folder is slow per stream but scales: read the weights in 16 streams first, so loading reads memory.
    bar.set_description_str('Reading Gemma')
    fd = os.open(weights / 'model.safetensors', os.O_RDONLY)
    offsets = range(0, os.fstat(fd).st_size, PIECE)

    def read(offset):
        os.pread(fd, PIECE, offset)
        bar.update(80 / len(offsets))

    with ThreadPoolExecutor(16) as pool:
        reading = pool.map(read, offsets)   # starts every read
        import torch
        from compressed_tensors.offload import remove_dispatch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(weights)
        list(reading)                       # waits for the reads, and raises if one failed
    os.close(fd)

    bar.n = 85
    bar.set_description_str('Loading Gemma')
    config = AutoConfig.from_pretrained(weights)
    config.quantization_config['dequantize'] = True   # unpacks the 4-bit weights to bfloat16 while loading
    with contextlib.redirect_stderr(io.StringIO()):
        model = AutoModelForImageTextToText.from_pretrained(weights, config=config, dtype=torch.bfloat16,
                                                            device_map='cuda')
    remove_dispatch(model)   # compressed-tensors leaves a wrapper on every layer, which slows every token down
    bar.n = 97

    bar.set_description_str('Installing Chromium')
    if chromium.wait():
        raise RuntimeError('Installing Chromium failed')
    bar.n = 100
    bar.set_description_str('Ready')
