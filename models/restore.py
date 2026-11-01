"""Download official Qwen3 repositories; resumable ranges and SHA-256 verification."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import struct
import threading
import time
from urllib.parse import quote

import requests
import urllib3.util.connection

# Scope IPv4 preference to this downloader; leave machine networking unchanged.
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / '.downloads'
CACHE = None
NAMES = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B']
CHUNK = 32 * 1024 * 1024
WORKERS = 32
PRINT_LOCK = threading.Lock()


def log(*args):
    with PRINT_LOCK:
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *args, flush=True)


def save_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def download_file(name, file):
    destination = ROOT / name / file['Path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    size, expected = file['Size'], file['Sha256']
    if destination.is_symlink():
        raise RuntimeError(f'Refusing symlink: {destination}')
    if destination.exists():
        if destination.stat().st_size == size and sha256(destination) == expected:
            log('VERIFIED_EXISTING', name, file['Path'])
            return
        raise RuntimeError(f'Existing file differs; preserved without overwrite: {destination}')

    candidate = CACHE / name / file['Path'] if CACHE else None
    if candidate is not None and candidate.is_file() and candidate.stat().st_size == size:
        copied = CONTROL / 'reuse' / name / file['Path']
        copied.parent.mkdir(parents=True, exist_ok=True)
        log('COPY_CACHED', name, file['Path'], size)
        shutil.copyfile(candidate, copied)
        if sha256(copied) == expected:
            os.replace(copied, destination)
            log('COPIED_VERIFIED', name, file['Path'], size, expected)
            return
        log('CACHE_HASH_DIFFERS', name, file['Path'], 'using network instead')

    # Reuse identical tokenizer/config files already verified for another size.
    if not file['Path'].endswith('.safetensors'):
        for other in NAMES:
            candidate = ROOT / other / file['Path']
            if other != name and candidate.is_file() and candidate.stat().st_size == size and sha256(candidate) == expected:
                shutil.copyfile(candidate, destination)
                log('REUSED_VERIFIED', name, file['Path'], 'from', other)
                return

    work = CONTROL / name / file['Path']
    work.parent.mkdir(parents=True, exist_ok=True)
    partial = work.with_name(work.name + '.part')
    state_path = work.with_name(work.name + '.parts.json')
    identity = {'sha256': expected, 'size': size, 'chunk_size': CHUNK}
    state = dict(identity, done=[])
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if any(state.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f'Partial download manifest changed: {state_path}')
    if state['done'] and (not partial.exists() or partial.stat().st_size != size):
        raise RuntimeError('Partial download data missing or wrong size')
    descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(descriptor, size)
    done = set(state['done'])
    state_lock = threading.Lock()
    last_report = [0.0]
    url = f"https://modelscope.cn/models/Qwen/{name}/resolve/{file['Revision']}/{quote(file['Path'], safe='/')}"
    urls = [url]
    chunks = (size + CHUNK - 1) // CHUNK

    def fetch(index):
        start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
        for attempt in range(6):
            try:
                started = time.monotonic()
                with requests.get(urls[attempt % len(urls)], headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, stream=True, timeout=(8, 20)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise RuntimeError('Incorrect Content-Range')
                    elif response.status_code != 200 or start != 0 or end + 1 != size:
                        raise RuntimeError('Server did not honor range request')
                    offset = start
                    for data in response.iter_content(256 * 1024):
                        if time.monotonic() - started > (180 if file['Path'].endswith('.safetensors') else 1800):
                            raise TimeoutError('Range transfer exceeded deadline')
                        if offset + len(data) > end + 1:
                            raise RuntimeError('Download exceeds requested range')
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(descriptor, view, offset)
                            if written <= 0:
                                raise OSError('Short disk write')
                            offset += written
                            view = view[written:]
                    if offset != end + 1:
                        raise RuntimeError('Incomplete range')
                with state_lock:
                    done.add(index)
                    state['done'] = sorted(done)
                    save_json(state_path, state)
                    now = time.monotonic()
                    if now - last_report[0] >= 20 or len(done) == chunks:
                        last_report[0] = now
                        downloaded = sum(min(CHUNK, size - number * CHUNK) for number in done)
                        log('PROGRESS', name, file['Path'], f'{downloaded}/{size}', f'{100*downloaded/max(size,1):.1f}%')
                return
            except Exception as error:
                log('RANGE_RETRY', name, file['Path'], index, attempt + 1, type(error).__name__)
                if attempt == 5:
                    raise
                time.sleep(min(30, 2 ** attempt))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(fetch, (index for index in range(chunks) if index not in done)))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if partial.stat().st_size != size or sha256(partial) != expected:
        raise RuntimeError(f'SHA-256 mismatch; partial preserved for inspection: {partial}')
    os.replace(partial, destination)
    log('VERIFIED', name, file['Path'], size, expected)


def validate_model(name, files):
    directory = ROOT / name
    config = json.loads((directory / 'config.json').read_text())
    assert config['model_type'] == 'qwen3'
    weights = {file['Path'] for file in files if file['Path'].endswith('.safetensors')}
    for index_path in directory.glob('*.safetensors.index.json'):
        index = json.loads(index_path.read_text())
        assert set(index['weight_map'].values()) <= weights
    tensor_count = 0
    for filename in sorted(weights):
        with (directory / filename).open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > 100 * 1024 * 1024:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(length))
            data_size = (directory / filename).stat().st_size - 8 - length
            for key, value in header.items():
                if key == '__metadata__':
                    continue
                first, last = value['data_offsets']
                assert 0 <= first <= last <= data_size
                tensor_count += 1
    tokenizer = json.loads((directory / 'tokenizer.json').read_text())
    tokenizer_config = json.loads((directory / 'tokenizer_config.json').read_text())
    assert tokenizer['model']['vocab'] and tokenizer_config['tokenizer_class']
    return {'status': 'complete', 'files': len(files), 'bytes': sum(file['Size'] for file in files), 'tensors': tensor_count}



def main():
    parser = argparse.ArgumentParser(description='Restore the exact four Qwen3 snapshots recorded on 2026-10-30 (Linux).')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cache-dir', type=Path, help='Optional directory containing Qwen3-* source folders; copies are hash-checked.')
    parser.add_argument('--models', nargs='+', choices=NAMES, default=NAMES)
    parser.add_argument('--workers', type=int, default=8, help='Concurrent 32 MiB ranges per file (1-32).')
    parser.add_argument('--verify-only', action='store_true', help='Hash and validate local files without downloading.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    global ROOT, CONTROL, CACHE, WORKERS
    ROOT = args.output.resolve()
    CONTROL = ROOT / '.downloads'
    CACHE = args.cache_dir.resolve() if args.cache_dir else None
    WORKERS = args.workers
    manifest_path = Path(__file__).resolve().with_name('manifest.json')
    if sha256(manifest_path) != '607913fbc0cc085ade8c34d810124c9307abe0ab08e5726c3d567650d8a2c0b9':
        raise ValueError('Frozen manifest differs from the recorded snapshot; refusing an implicit version change.')
    manifest = json.loads(manifest_path.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'restore.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = {name: manifest[name] for name in args.models}
        total = sum(file['Size'] for model in selected.values() for file in model['files'])
        remaining = sum(file['Size'] for name, model in selected.items() for file in model['files'] if not (ROOT / name / file['Path']).exists())
        for name, model in selected.items():
            for file in model['files']:
                partial = (CONTROL / name / file['Path']).with_name(Path(file['Path']).name + '.part')
                if not (ROOT / name / file['Path']).exists() and partial.is_file():
                    remaining -= min(file['Size'], partial.stat().st_blocks * 512)
        remaining = max(0, remaining)
        if not args.verify_only and shutil.disk_usage(ROOT).free < remaining + 2 * 1024 ** 3:
            raise OSError('Insufficient free filesystem space')
        log('VERIFY' if args.verify_only else 'RESTORE', 'bytes', total, 'workers', WORKERS)
        report = {}
        for name, model in selected.items():
            files = model['files']
            for file in sorted(files, key=lambda file: (not file['Path'].endswith('.safetensors'), file['Path'])):
                if args.verify_only:
                    path = ROOT / name / file['Path']
                    if not path.is_file() or path.stat().st_size != file['Size'] or sha256(path) != file['Sha256']:
                        raise ValueError(f'Missing or mismatched file: {path}')
                    log('VERIFIED', name, file['Path'])
                else:
                    download_file(name, file)
            report[name] = validate_model(name, files)
            save_json(CONTROL / 'restore-verification.json', report)
            log('MODEL_COMPLETE', name, json.dumps(report[name]))
        log('ALL_MODELS_COMPLETE', len(report), total)


if __name__ == '__main__':
    main()
