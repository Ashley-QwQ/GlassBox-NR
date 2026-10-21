#!/usr/bin/env python3
"""Run the 1080p frame-0 chain (158 launches) from an unpacked M2 payload and check it against the reference table.

  python run_1080_f0.py --root R --payload P --weights W --tables T --inputs I [--threads N] [--verify-only]

  R        work directory (receives work/ = relocated configuration, layer/, out/)
  P        the unpacked payload: the directory that holds kit/data/... and kit/home/... of the release
  W        directory with the weight files made from your own copy of the vendor library (names as in data/manifest.json)
  T        directory in which the release table volumes were extracted (it contains tables/<sha256><ext>)
  I        directory with the frozen frame-0 input files (names as in data/manifest.json)
The driver (1) relocates the configuration files to R, (2) checks every external file by size and sha256, (3) runs
devkit_worker.py, (4) compares the workspace and texture hashes of every launch and the final image with data/reference_f0.json.
Exit code 0 = every launch and the final image match.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import relocate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--payload', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tables', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--stop-after', type=int, default=157)
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    payload = os.path.abspath(a.payload)
    manifest = json.load(open(os.path.join(KIT, 'data', 'manifest.json')))
    ref = json.load(open(os.path.join(KIT, 'data', 'reference_f0.json')))
    t0 = time.time()
    os.makedirs(root, exist_ok=True)
    relocate.run(root, payload, manifest, {'weights': a.weights, 'tables': a.tables, 'inputs': a.inputs})
    rel = json.load(open(os.path.join(root, 'RELOCATED.json')))
    lay = manifest['layer']
    layer = os.path.join(root, 'layer')
    os.makedirs(os.path.join(layer, 'runtime'), exist_ok=True)
    work = root + '/work'
    for name, target in (('repo', work + lay['repo']), ('build', payload + lay['build'])):
        link = os.path.join(layer, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
    shutil.copyfile(rel['root_config'], os.path.join(layer, 'runtime/board-config.json'))
    shutil.copyfile(os.path.join(KIT, 'bridge.py'), os.path.join(layer, 'runtime/bridge.py'))
    job = {'layer': layer, 'inputs': work + manifest['job']['inputs'], 'ex2': work + manifest['job']['ex2'],
           'fill': 0, 'workspace_bytes': manifest['job']['workspace_bytes'], 'threads': a.threads,
           'output': os.path.join(root, 'out'), 'bridge_name': 'bridge.py', 'stop_after': a.stop_after}
    os.makedirs(job['output'], exist_ok=True)
    jp = os.path.join(root, 'job.json')
    json.dump(job, open(jp, 'w'), indent=1)
    if a.verify_only:
        print('VERIFY_ONLY: relocation and external-file checks passed (%d s)' % (time.time() - t0))
        return 0
    rc = subprocess.call([sys.executable, '-B', os.path.join(KIT, 'devkit_worker.py'), jp])
    if rc != 0:
        print('RUN_FAILED rc=%d (see out/first-failure.json)' % rc)
        return rc
    got = json.load(open(os.path.join(job['output'], 'perlaunch.json')))
    bad = None
    for g in got:
        r = ref['launches'][g['L']]
        if g['workspace_sha256'] != r['workspace_sha256'] or g['textures'] != r['textures']:
            bad = g['L']
            break
    final = json.load(open(os.path.join(job['output'], 'result.json')))['final_sha256']
    if bad is not None:
        print('MISMATCH at launch', bad)
        return 2
    print('ALL %d LAUNCHES MATCH THE REFERENCE TABLE' % len(got))
    if a.stop_after >= 157:
        print('FINAL IMAGE', final, 'MATCH' if final == ref['final_sha256'] else 'DIFFERENT')
        return 0 if final == ref['final_sha256'] else 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
