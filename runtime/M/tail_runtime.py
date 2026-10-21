"""C4 per-stage final-owner descriptors with sealed E1 compact LG2 assets."""
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '4255d067e6a618342182a86f44fbf2feceb92a7d148e0dcdfaabe304f8c72138'

def safe(path):
    path = Path(path)
    if (''.join(map(chr,(98,108,105,110,100,))) in str(path).lower()):
        raise ValueError('forbidden path')
    return path

def stage_tables(config, launch):
    if launch not in (155, 156, 157):
        raise ValueError('tail launch')
    if launch == 156:
        return {}
    root = safe(config['work']) / 'tables'
    manifest = json.loads(safe(root / 'manifest.json').read_bytes())
    result = dict(manifest['runtime'])
    for key in (('rcp', 'rsq') if launch == 155 else ('ex2', 'rcp')):
        pin = manifest['sources'][key]
        result[key] = {'path': str(safe(root / (key + '.bin'))),
                       'bytes': pin['bytes'], 'sha256': pin['extracted_sha256']}
    if launch == 157:
        encoded = safe(config['e1_encoded'])
        raw = safe(encoded / 'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
            raise ValueError('E1 manifest identity')
        codec = json.loads(raw)
        if codec['endian'] != 'little':
            raise ValueError('E1 endian')
        if result['lg2_blocks'] != list(range(111, 128)):
            raise ValueError('LG2 size')
        result['encoded_manifest'] = raw
        for key, filename in [('lg2_base', 'base_E127.bin'),
                              ('lg2_e126', 'raw_E126.bin'),
                              ('lg2_levels', 'level_3bit.bin')]:
            result[key] = {'path': str(safe(encoded / filename)), **codec['files'][filename]}
    return result
