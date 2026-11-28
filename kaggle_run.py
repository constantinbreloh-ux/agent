"""Builds the notebook, runs it on Kaggle with the starter's inputs and GPU, and downloads its output.

python kaggle_run.py           build, push as a new version, wait, download output to runs/<version>/
python kaggle_run.py --fetch   only wait for and download the latest version
python kaggle_run.py --wheels  build the vLLM wheels once (CPU, internet on), into the notebook gga-vllm-wheels

Auth: the session proxy injects the real Kaggle token; the token below is only a placeholder.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest, ApiSaveKernelRequest)

USER = 'constantinbreloh'
SLUG = 'gga-autopilot'
TEMPLATE = 'gga-starter-gemma-4-31b-on-transformers'   # copy inputs, GPU and image from this notebook
WHEELS = 'gga-vllm-wheels'                            # its output is attached as an input
here = Path(__file__).parent
api = KaggleClient(api_token=os.getenv('KAGGLE_API_TOKEN', 'KGAT_proxyinjected00000000000000')).kernels.kernels_api_client


def request(cls, **fields):
    r = cls()
    for k, v in fields.items():
        setattr(r, k, v)
    return r


def push():
    subprocess.run([sys.executable, str(here / 'make_notebook.py')], check=True)
    t = api.get_kernel(request(ApiGetKernelRequest, user_name=USER, kernel_slug=TEMPLATE)).metadata
    r = api.save_kernel(request(
        ApiSaveKernelRequest, slug=f'{USER}/{SLUG}', new_title='GGA AutoPILOT', text=(here / 'gga_autopilot.ipynb').read_text(),
        language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=False,
        machine_shape=t.machine_shape, docker_image=t.docker_image, competition_data_sources=t.competition_data_sources,
        model_data_sources=t.model_data_sources, dataset_data_sources=t.dataset_data_sources,
        kernel_data_sources=[f'{USER}/{WHEELS}']))
    if r.error or r.invalid_model_sources or r.invalid_competition_sources or r.invalid_kernel_sources:
        sys.exit(f'push failed: {r.error} {r.invalid_model_sources} {r.invalid_competition_sources} {r.invalid_kernel_sources}')
    print(f'pushed version {r.version_number}: {r.url}')
    return r.version_number


def push_wheels():
    t = api.get_kernel(request(ApiGetKernelRequest, user_name=USER, kernel_slug=TEMPLATE)).metadata
    r = api.save_kernel(request(
        ApiSaveKernelRequest, slug=f'{USER}/{WHEELS}', new_title='GGA vLLM wheels', text=(here / 'notebook/wheels.py').read_text(),
        language='python', kernel_type='script', is_private=True, enable_gpu=False, enable_internet=True,
        docker_image=t.docker_image))   # same image as the GPU notebook, so the wheels match its Python
    if r.error:
        sys.exit(f'push failed: {r.error}')
    print(f'pushed version {r.version_number}: {r.url}')
    return r.version_number


def wait(slug=SLUG):
    last = None
    while True:
        s = api.get_kernel_session_status(request(ApiGetKernelSessionStatusRequest, user_name=USER, kernel_slug=slug))
        status = str(s.status).split('.')[-1]
        if status != last:
            print(time.strftime('%H:%M:%S'), status, s.failure_message or '')
            last = status
        if status not in ('QUEUED', 'RUNNING', 'NEW_SCRIPT'):
            return status
        time.sleep(30)


def fetch(version, slug=SLUG):
    out = here / 'runs' / (str(version) if slug == SLUG else f'{slug}-{version}')
    out.mkdir(parents=True, exist_ok=True)
    r = api.list_kernel_session_output(request(ApiListKernelSessionOutputRequest, user_name=USER, kernel_slug=slug))
    (out / 'log.txt').write_text(r.log or '')
    for f in r.files:
        if f.file_name.endswith(('.txt', '.parquet', '.json', '.py', '.webm', '.log')):
            (out / f.file_name).parent.mkdir(parents=True, exist_ok=True)   # videos sit in one folder per game
            urllib.request.urlretrieve(f.url, out / f.file_name)
    print(f'output in {out}: {sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())}')


if __name__ == '__main__':
    if '--wheels' in sys.argv:
        version = push_wheels()
        wait(WHEELS)
        fetch(version, WHEELS)
    else:
        version = 'latest' if '--fetch' in sys.argv else push()
        wait()
        fetch(version)
