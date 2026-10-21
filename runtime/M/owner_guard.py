"""Reject shared backing and overlapping writable/read-only buffer ranges before snapshots."""
import ctypes
class Buffer(ctypes.Structure):
    _fields_=[('buf',ctypes.c_void_p),('obj',ctypes.c_void_p),('length',ctypes.c_ssize_t),('itemsize',ctypes.c_ssize_t),('readonly',ctypes.c_int),('ndim',ctypes.c_int),('format',ctypes.c_char_p),('shape',ctypes.POINTER(ctypes.c_ssize_t)),('strides',ctypes.POINTER(ctypes.c_ssize_t)),('suboffsets',ctypes.POINTER(ctypes.c_ssize_t)),('internal',ctypes.c_void_p)]
get=ctypes.pythonapi.PyObject_GetBuffer;get.argtypes=[ctypes.py_object,ctypes.POINTER(Buffer),ctypes.c_int];get.restype=ctypes.c_int
release=ctypes.pythonapi.PyBuffer_Release;release.argtypes=[ctypes.POINTER(Buffer)];release.restype=None

def backing(owner):
    seen=set()
    while id(owner) not in seen:
        seen.add(id(owner))
        if isinstance(owner,memoryview):owner=owner.obj
        elif getattr(owner,'base',None) is not None:owner=owner.base
        else:return owner
    raise RuntimeError('BUFFER_BACKING_CYCLE')

def extent(owner):
    b=Buffer()
    try:get(owner,ctypes.byref(b),0x11c)
    except TypeError:return None
    try:
        low=high=int(b.buf or 0)
        if b.length:
            for i in range(b.ndim):
                stride=b.strides[i] if b.strides else b.itemsize
                span=(b.shape[i]-1)*stride
                low+=min(0,span);high+=max(0,span)
            high+=b.itemsize
        return low,high
    finally:release(ctypes.byref(b))

def reject_aliases(L,workspace,textures):
    outputs={155:{9},156:{6},157:{2}}.get(L,set())
    owners=[('workspace',workspace,True)]+[(f'texture{k}',v['data'] if isinstance(v,dict) else v,k in outputs) for k,v in textures.items() if isinstance(k,int)]
    spans=[(name,id(backing(v)),extent(v),w) for name,v,w in owners]
    for i,(a,ar,ab,aw) in enumerate(spans):
        for b,br,bb,bw in spans[i+1:]:
            if not (aw or bw):continue
            if ar==br or (ab is not None and bb is not None and max(ab[0],bb[0])<min(ab[1],bb[1])):raise RuntimeError('WRITABLE_BORROW_ALIAS:'+a+':'+b)
    return spans
