#!/usr/bin/env python3
"""Lay the content-addressed payload archive out the way the devkit expects (no numeric code involved).

The archive (payload.tar.zst.NNN) extracts to  payload/<sha256><ext>.  PAYLOAD_MANIFEST.json maps every shipped path to its sha256
and member.  This step puts every file at its place:
  * root PAYLOAD entries   -> <out>/<canonical path>                 (the directory the devkit calls --payload)
  * release_asset entries  -> <tables>/tables/<sha256><ext>          (the directory the devkit calls --tables)
  * entries that live in git (python_runtime, compiled_module, launch_params; where == "git") are taken from the git checkout
    (--repo, resolved through docs/FILES_MAP_V2.json: whitelist path -> repo path) or, as before, from <githead>/PAYLOAD/<path>
Files are hard-linked from the extraction (copied if linking fails), sizes are checked, git files are checked by sha256.

usage: payload_layout.py --manifest PAYLOAD_MANIFEST.json --extracted DIR (--repo GIT_CHECKOUT | --githead DIR) --out PAYLOAD_DIR --tables TABLES_DIR
  DIR (extracted) is the directory that contains payload/ ; GIT_CHECKOUT is this repository (needs docs/FILES_MAP_V2.json);
  githead (legacy) has PAYLOAD/<canonical path> for the git categories.
  --check-map only verifies that every git-resident manifest entry resolves to a file of the right size and sha256, and exits
  (needs --manifest and --repo only).
"""
import argparse, hashlib, json, os, shutil, sys


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return 'link'
    except OSError:
        shutil.copyfile(src, dst)
        return 'copy'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('manifest', 'extracted', 'githead', 'repo', 'out', 'tables'):
        ap.add_argument('--' + k)
    ap.add_argument('--check-map', action='store_true')
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    repo_index = None
    if a.repo:
        fm = json.load(open(os.path.join(a.repo, 'docs', 'FILES_MAP_V2.json')))
        repo_index = {}
        for f in fm['files']:
            for wp in f['whitelist_paths']:
                repo_index[wp] = os.path.join(a.repo, *f['repo_path'].split('/'))
    if a.check_map:
        if repo_index is None:
            raise SystemExit('--check-map needs --repo')
        n = 0
        for e in man['files']:
            if e['root'] == 'PAYLOAD' and e['where'] == 'git':
                src = repo_index.get(e['path'])
                if src is None or not os.path.isfile(src):
                    raise SystemExit('GIT_FILE_MISSING ' + e['path'])
                if os.path.getsize(src) != e['bytes'] or sha_file(src) != e['sha256']:
                    raise SystemExit('GIT_FILE_SHA_DIFF ' + e['path'])
                n += 1
        print('file map OK: %d git-resident entries resolve and match sha256' % n)
        return 0
    for k in ('extracted', 'out', 'tables'):
        if not getattr(a, k):
            raise SystemExit('missing --' + k)
    if not (a.repo or a.githead):
        raise SystemExit('need --repo or --githead')
    out, tables = os.path.abspath(a.out), os.path.abspath(a.tables)
    n = {'link': 0, 'copy': 0, 'git': 0, 'tables': 0}
    for e in man['files']:
        if e['root'] == 'RELEASE_VOLUMES':
            src, dst = os.path.join(a.extracted, e['member']), os.path.join(tables, e['path'])
        elif e['root'] == 'PAYLOAD' and e['where'] == 'payload':
            src, dst = os.path.join(a.extracted, e['member']), out + e['path']
        elif e['root'] == 'PAYLOAD':
            if repo_index is not None:
                src = repo_index.get(e['path'])
                if src is None:
                    raise SystemExit('GIT_FILE_NOT_IN_MAP ' + e['path'])
            else:
                src = os.path.join(a.githead, 'PAYLOAD') + e['path']
            dst = out + e['path']
            if sha_file(src) != e['sha256']:
                raise SystemExit('GIT_FILE_SHA_DIFF ' + e['path'])
            n['git'] += 1
        else:
            continue                                            # M2SRC / DEVKIT: not part of the run layout
        if os.path.getsize(src) != e['bytes']:
            raise SystemExit('SIZE_DIFF ' + e['path'])
        n[put(src, dst)] += 1
        if e['root'] == 'RELEASE_VOLUMES':
            n['tables'] += 1
    print('layout done', n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
