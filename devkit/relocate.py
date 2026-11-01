#!/usr/bin/env python3
"""Relocate an M2 payload to the place where it was unpacked.

The shipped configuration files keep the path layout of the sealed runs: every path string starts with one of ROOTS
(/kit/data/ or /kit/home/).  The payload directory mirrors that layout (payload/kit/data/..., payload/kit/home/...).
This tool writes a relocated copy of the configuration files named in the manifest into <root>/work and re-pins them:
  * a path string that names a rewritten file maps to <root>/work/..., one that names an external input maps to the file the
    user supplied (checked by size and sha256), any other existing payload file maps to <payload>/...;
  * every {path, bytes, sha256} record whose target was rewritten gets the new size and sha256 (dependency order);
  * lists of such records that were sorted by path stay sorted;
  * link names (a shipped name that is a link to another shipped file or directory) become links in <root>/work;
  * directory mirrors (configuration strings that name a directory) are laid out under <root>/work;
  * the derived map and the pinned derived_pins.py of the loader directory are regenerated for the relocated shas, and the
    execution seal's pins for these two files are refreshed (same order of steps as the build that produced the payload).
Payload files that are not rewritten are used where they are.

usage: relocate.py <root> <payload_dir> <manifest.json> role=dir [role=dir ...]
  manifest: rewrite [path], copy [path], aliases {path: path}, dir_mirrors [path], external [{role,name,path,bytes,sha256}],
            loader_dir, map_path, pins_path, seal_path, root_config           (all canonical absolute paths)
"""
import hashlib
import json
import os
import re
import shutil
import sys

ROOTS = ('/kit/data/', '/kit/home/')
REL_OK = re.compile(r'^[\w.+\-][\w.+\-/]{2,200}$')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run(root, payload, manifest, ext_dirs):
    root = os.path.abspath(root)
    payload = os.path.abspath(payload)
    W = root + '/work'
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plan_format as pf            # the shipped codec of the plan indexes (next to this file)
    sys.setrecursionlimit(100000)

    external = {}
    for e in manifest['external']:
        d = ext_dirs.get(e['role'])
        if d is None:
            if e['role'] in manifest.get('optional_external_roles', ()):
                continue                         # e.g. inputs_f1: only the two-frame driver supplies them
            raise SystemExit('EXTERNAL_ROLE_NOT_GIVEN role=' + e['role'])
        p = os.path.join(d, e['name'])
        if not os.path.isfile(p) or os.path.getsize(p) != e['bytes'] or sha_file(p) != e['sha256']:
            raise SystemExit('EXTERNAL_FILE_MISSING_OR_DIFFERENT role=%s name=%s' % (e['role'], e['name']))
        external[e['path']] = os.path.abspath(p)

    rewrite = set(manifest['rewrite'])
    copies = set(manifest.get('copy', []))
    aliases = dict(manifest.get('aliases', {}))
    mirrors = set(d.rstrip('/') for d in manifest.get('dir_mirrors', []))
    generated = {manifest['map_path'], manifest['pins_path']}
    seal_path = manifest['seal_path']
    exists_memo = {}

    def lexists(p):
        v = exists_memo.get(p)
        if v is None:
            v = exists_memo[p] = os.path.lexists(p)
        return v

    def real_of(p):
        """resolve link names (a link directory in front of the path, or the path itself)"""
        for _ in range(8):
            for a, t in aliases.items():
                if p == a:
                    p = t
                    break
                if p.startswith(a + '/'):
                    p = t + p[len(a):]
                    break
            else:
                return p
        return p

    def new_path(p):
        if not p.startswith(ROOTS):
            return None
        r = real_of(p)
        if r != p:
            return W + p                      # a link name: the link lives in <root>/work
        if r in rewrite or r in external or r in copies or r in generated or r in mirrors:
            return W + r
        q = payload + r
        return q if lexists(q) else None

    def map_str(s, base):
        if s.startswith(ROOTS):
            return new_path(s) or s
        if base and REL_OK.match(s) and any(ch in s for ch in './_-'):
            c = os.path.normpath(os.path.join(base, s))
            if c.startswith(ROOTS) and lexists(payload + real_of(c)):
                return new_path(c) or s
        return s

    def write(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + '.tmp', 'wb') as f:
            f.write(data)
        os.replace(path + '.tmp', path)

    memo = {}
    in_sha = {}          # canonical path -> sha256 of the shipped input
    out_ident = {}       # canonical path -> identity of the relocated output

    def produce(p):
        """identity of the relocated version of canonical path p (None if the file is used unchanged)"""
        r = real_of(p)
        if r in memo:
            return memo[r]
        memo[r] = None
        if r not in rewrite:
            return None
        raw = open(payload + r, 'rb').read()
        in_sha[r] = sha_bytes(raw)
        if raw[:8] == b'MRGPLAN1':
            obj, is_plan = pf.decode(raw[8:]), True
        else:
            obj, is_plan = json.loads(raw.decode('utf-8-sig')), False
        out = rebind(obj, os.path.dirname(r))
        data = (b'MRGPLAN1' + pf.encode(out)) if is_plan else (json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
        write(W + r, data)
        memo[r] = out_ident[r] = {'bytes': len(data), 'sha256': sha_bytes(data)}
        return memo[r]

    def rebind(o, base, key=None):
        if isinstance(o, dict):
            tgt = o.get('path') if isinstance(o.get('path'), str) else None
            out = {(map_str(k, None) if isinstance(k, str) else k): rebind(v, base, k) for k, v in o.items()}
            if tgt and 'sha256' in o:
                if not tgt.startswith(ROOTS) and REL_OK.match(tgt):
                    tgt = os.path.normpath(os.path.join(base, tgt))
                if tgt.startswith(ROOTS) and real_of(tgt) not in generated:
                    ident = produce(tgt)
                    if ident is not None:
                        out['sha256'] = ident['sha256']
                        if 'bytes' in o:
                            out['bytes'] = ident['bytes']
            return out
        if isinstance(o, list):
            res = [rebind(v, base) for v in o]
            if key == 'assets' and len(o) > 1 and all(isinstance(v, dict) and isinstance(v.get('path'), str) and 'sha256' in v for v in o):
                if [v['path'] for v in o] == sorted(v['path'] for v in o):
                    res.sort(key=lambda v: v['path'])
            return res
        if isinstance(o, str):
            return map_str(o, base)
        return o

    # link names: <root>/work/<name> -> the (possibly relocated) target
    for a, t in sorted(aliases.items(), key=lambda kv: len(kv[0])):
        os.makedirs(os.path.dirname(W + a), exist_ok=True)
        if os.path.lexists(W + a):
            os.remove(W + a)
        os.symlink(new_path(t) or t, W + a)
    # directory mirrors (directory strings in the configuration): <root>/work/<dir> holds the rewritten files of that directory as
    # real files and a link to the shipped file or directory for every other entry
    def mirror_into(src, t):
        os.makedirs(t, exist_ok=True)
        if not os.path.isdir(src):
            return
        for name in sorted(e.name for e in os.scandir(src) if ''.join(map(chr,(98,108,105,110,100))) not in e.name.lower()):
            sp, dp = src + '/' + name, t + '/' + name
            if os.path.lexists(dp):
                if os.path.isdir(dp) and not os.path.islink(dp) and os.path.isdir(sp):
                    mirror_into(sp, dp)               # a directory that already holds real files: fill in the rest
                continue
            os.symlink(sp, dp)

    def mirror_dir(d):
        mirror_into(payload + real_of(d), W + d)

    pending_mirrors = sorted(mirrors)

    for p in sorted(rewrite):
        produce(p)
    for p, src in sorted(external.items()):      # the user's files are installed (hard link, else copy) at the name the configuration uses
        dst = W + p
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    for p in sorted(copies):                     # loader modules: byte copies, so that the loader directory is one directory
        write(W + p, open(payload + p, 'rb').read())
    for d in pending_mirrors:                     # after the real files exist: link the other entries
        mirror_dir(d)

    # derived map and pinned derived_pins.py for the relocated shas
    base_map = json.load(open(payload + manifest['map_path'], encoding='utf-8'))
    by_orig = {}
    for e in base_map['entries']:
        assert e['original_sha256'] not in by_orig, 'DERIVED_MAP_DUPLICATE_ORIGINAL in the shipped map'
        by_orig[e['original_sha256']] = dict(e)
    by_derived = {}
    for e in by_orig.values():
        by_derived.setdefault(e['derived_sha256'], []).append(e)
    for p, ident in sorted(out_ident.items()):
        r = in_sha[p]
        if r == ident['sha256']:
            continue
        if r in by_derived:                   # the shipped file was itself a derived file: the original now maps to the relocated one
            for e in by_derived[r]:
                e['derived_sha256'], e['derived_bytes'] = ident['sha256'], ident['bytes']
        else:                                  # the shipped file is an original of the sealed runs
            e = by_orig.setdefault(r, {'original_sha256': r, 'original_bytes': os.path.getsize(payload + p), 'kind': 'relocated'})
            e['derived_sha256'], e['derived_bytes'] = ident['sha256'], ident['bytes']
    entries = sorted(by_orig.values(), key=lambda e: e['original_sha256'])
    doc = json.dumps({'schema': 'DERIVED_MAP/1', 'entries': entries}, separators=(',', ':')).encode('utf-8')
    msha = sha_bytes(doc)
    old_msha = sha_file(payload + manifest['map_path'])
    pins_src = open(payload + manifest['pins_path'], encoding='utf-8').read()
    assert old_msha in pins_src, 'DERIVED_MAP_SHA256 constant not found in derived_pins.py'
    write(W + manifest['map_path'], doc)
    write(W + manifest['pins_path'], pins_src.replace(old_msha, msha, 1).encode('utf-8'))

    # the execution seal pins both generated files
    sealp = W + seal_path
    seal = json.load(open(sealp, encoding='utf-8'))
    gen = (W + manifest['map_path'], W + manifest['pins_path'])
    for q in seal['pins']:
        if q['path'] in gen:
            q['bytes'], q['sha256'] = os.path.getsize(q['path']), sha_file(q['path'])
    write(sealp, (json.dumps(seal, separators=(',', ':')) + '\n').encode('utf-8'))

    json.dump({'root': root, 'payload': payload, 'rewritten': len(out_ident), 'external': {k: W + k for k in external},
               'root_config': new_path(manifest['root_config']), 'seal': sealp,
               'loader_override_dir': os.path.dirname(W + manifest['map_path']), 'map_sha256': msha},
              open(root + '/RELOCATED.json', 'w', encoding='utf-8'), indent=1)
    print('relocated', len(out_ident), 'files; derived map entries', len(entries))


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), dict(a.split('=', 1) for a in sys.argv[4:]))
