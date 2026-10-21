"""G0b bounded signed-map loader and original-representation canonical bridge."""
import hashlib, json, math, os, zipfile
from pathlib import Path
import numpy as np
CHUNK = 256 << 10
STRATEGY = {k:'<i4' for k in ('wload_off','sread_src','gstore_off','xsite_src','xsite_wsrc')}
STRATEGY.update(shfl_idx='|i1', psel='|i1', thread_class='|i1')
def safe(p):
 p=Path(p)
 if (''.join(map(chr,(98,108,105,110,100,))) in str(p).lower()):raise RuntimeError('FORBIDDEN_PATH')
 return p

def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 p=safe(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def header(f):
 v=np.lib.format.read_magic(f)
 if v==(1,0):shape,fort,dtype=np.lib.format.read_array_header_1_0(f)
 elif v==(2,0):shape,fort,dtype=np.lib.format.read_array_header_2_0(f)
 else:raise ValueError('NPY_VERSION_REJECTED')
 if fort or dtype.kind!='i' or not dtype.isnative or dtype.itemsize not in (1,2,4,8):raise ValueError('MAP_SIGNED_NATIVE_C_ORDER_REQUIRED')
 return shape,dtype

def read_narrow(certificate_path, expected_source=None, fields=None):
 cert=json.loads(safe(certificate_path).read_text(encoding='utf-8'))
 if cert['schema']!='G0B_STRATEGY_B_CERT_V1' or cert['all_value_diff_count']!=0:raise RuntimeError('MAP_CERTIFICATE_REJECTED')
 if expected_source is not None and cert['source']['sha256']!=expected_source:raise RuntimeError('MAP_SOURCE_BINDING_DIFF')
 npz=safe(safe(certificate_path).parent/cert['narrow']['relative'])
 before=npz.stat()
 if before.st_size!=cert['narrow']['bytes'] or sha(npz)!=cert['narrow']['sha256']:raise RuntimeError('NARROW_MAP_SHA_DIFF')
 arrays={}
 with zipfile.ZipFile(npz) as z:
  names=z.namelist()
  if any(''.join(map(chr,(98,108,105,110,100,))) in n.lower() for n in names):raise RuntimeError('FORBIDDEN_MEMBER')
  if sorted(names)!=sorted(k+'.npy' for k in cert['fields']) or len(set(names))!=len(names):raise RuntimeError('NARROW_MEMBERS_DIFF')
  for name,rec in cert['fields'].items():
   with z.open(name+'.npy') as f:
    shape,dtype=header(f)
    if list(shape)!=rec['shape'] or dtype.str!=rec['narrow_dtype']:raise RuntimeError('NARROW_DTYPE_SHAPE_DIFF')
    count=math.prod(shape)
    if z.getinfo(name+'.npy').file_size-f.tell()!=count*dtype.itemsize:raise RuntimeError('NARROW_MEMBER_LENGTH_DIFF')
    selected=fields is None or name in fields
    a=np.empty(shape,dtype=dtype,order='C') if selected else None
    view=memoryview(a).cast('B') if selected and count else None
    h=hashlib.sha256();done=0
    while done<count*dtype.itemsize:
     n=min(CHUNK,count*dtype.itemsize-done);b=f.read(n)
     if len(b)!=n:raise RuntimeError('NARROW_MEMBER_TRUNCATED')
     h.update(b)
     if view is not None:view[done:done+n]=b
     done+=n
    if f.read(1):raise RuntimeError('NARROW_MEMBER_EXTRA_DATA')
    if h.hexdigest()!=rec['narrow_payload_sha256']:raise RuntimeError('NARROW_PAYLOAD_DIFF')
    if view is not None:view.release()
    if selected:arrays[name]=a
 after=npz.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('NARROW_MAP_CHANGED')
 return arrays,cert

def stream_hash(a, dtype=None):
 """At most CHUNK input bytes plus CHUNK canonical bytes; no whole-member copy."""
 a=np.asarray(a);dst=np.dtype(dtype) if dtype is not None else a.dtype;h=hashlib.sha256()
 n=max(1,CHUNK//max(a.itemsize,dst.itemsize))
 # nditer also preserves C-order hashing for non-contiguous legacy arrays.
 it=np.nditer(a,flags=['external_loop','buffered','zerosize_ok'],op_flags=['readonly'],order='C',buffersize=n)
 for part in it:
  for start in range(0,part.size,n):
   b=np.ascontiguousarray(part[start:start+n],dtype=dst)
   h.update(memoryview(b).cast('B'))
 return h.hexdigest()
def canonical_fields(spec):
 records=spec.get('_g0b_fields');result={}
 for name,array in sorted(spec['maps'].items()):
  a=np.asarray(array)
  if records is None:
   result[name]=dict(shape=list(a.shape),strides=list(a.strides),dtype=a.dtype.str,sha256=stream_hash(a))
  else:
   rec=records[name]
   if a.dtype.str!=rec['narrow_dtype'] or list(a.shape)!=rec['runtime_shape']:raise RuntimeError('CANONICAL_MAP_CONTRACT_DIFF')
   logical=stream_hash(a,rec['source_dtype'])
   if logical!=rec['source_payload_sha256']:raise RuntimeError('CANONICAL_LOGICAL_VALUE_DIFF')
   result[name]=dict(shape=rec['runtime_shape'],strides=rec['source_runtime_strides'],dtype=rec['source_dtype'],sha256=logical)
 return result
