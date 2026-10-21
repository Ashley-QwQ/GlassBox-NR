"""Loader for frozen non-numeric maps; no checkpoint or target access."""
import json, hashlib
import derived_pins
from pathlib import Path
import numpy as np
from g0b_maps import read_narrow, canonical_fields
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

REQUIRED_FIELDS = ("wload_off", "sread_src", "gstore_off")
OPTIONAL_FIELDS = ("xsite_src", "xsite_wsrc", "shfl_idx", "psel", "thread_class")
def required_fields(wiring):
 kinds = {n[0] for n in wiring['nodes']}
 fields = set(REQUIRED_FIELDS)
 if kinds & {'X','XH'}: fields.update(('xsite_src','xsite_wsrc'))
 if 'SI' in kinds: fields.add('shfl_idx')
 if 'PS' in kinds: fields.add('psel')
 if 'classes' in wiring: fields.add('thread_class')
 return fields
def selected_maps(z,wiring):
 need = required_fields(wiring)
 names = set(z) if isinstance(z,dict) else set(z.files)
 missing = need - names
 if missing: raise ValueError('missing required map fields:'+','.join(sorted(missing)))
 return {k:np.ascontiguousarray(z[k]) for k in REQUIRED_FIELDS+OPTIONAL_FIELDS if k in names}

def load(config,launch):
 bundle=Path(config['work'])/'bundle';manifest=read_json(bundle/'manifest.json')
 row=next(r for r in manifest['rows'] if r['launch']==launch);sp=dict(row)
 for k in ('params','map','wiring','contract'):
  p=bundle/row[k]
  expected=row['params_sha256'] if k=='params' else row['sources'][{'map':0,'wiring':1,'contract':2}[k]]['sha256']
  # Bundle is pinned as a whole by driver fingerprint. Each source SHA is
  # checked directly here in original map/wiring/contract order.
  if k!='params': expected=row['sources'][{'map':0,'wiring':1,'contract':2}[k]]['sha256']
  if not derived_pins.accepts_sha(expected,sha(p)):raise RuntimeError('bundle SHA mismatch:'+str(p))
 sp['params']=(bundle/row['params']).read_bytes()
 sp['wiring']=read_json(bundle/row['wiring']);con=read_json(bundle/row['contract']);sp['contract']=con.get('contract',con)
 if 'g0b_certificates' in config:
  arrays,cert=read_narrow(config['g0b_certificates'][str(launch)],row['sources'][0]['sha256'],REQUIRED_FIELDS+OPTIONAL_FIELDS)
  sp['maps']=selected_maps(arrays,sp['wiring'])
  sp['_g0b_fields']=cert['fields']
 else:
  with np.load(bundle/row['map'],allow_pickle=False) as z:sp['maps']=selected_maps(z,sp['wiring'])
 if 'thread_class' in sp['maps']:sp['maps']['thread_class']=sp['maps']['thread_class'][...,None]
 tables={}
 for k in ('rsq','rcp'):
  with np.load(bundle/f'{k}-domain.npz',allow_pickle=False) as z:
   bits=z['input_half_bits']; vals=z['native_float_bits']; ix=bits.astype(np.int64)-0x410
   t=np.zeros(0x7c00-0x410,np.uint32);live=(ix>=0)&(ix<t.size);t[ix[live]]=vals[live];tables[k]=t.view(np.float32)
 return manifest,row,sp,tables


def normalized_key(spec, task_identity, epoch):
 """Identity only: every configure remains uncached and owns its arrays."""
 maps = canonical_fields(spec)
 payload = {'task_identity':str(task_identity),'epoch':int(epoch),'maps':maps,
  'wiring':spec['wiring'],'contract':spec['contract'],
  'params_sha256':hashlib.sha256(spec['params']).hexdigest()}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def require_normalized_key(spec, task_identity, epoch, expected):
 actual = normalized_key(spec, task_identity, epoch)
 if actual != expected:raise RuntimeError('NORMALIZED_SPLIT_IDENTITY_KEY_DIFF')
 return actual
