"""IG-a v1 portable little-endian assets; no native-object serialization."""
import hashlib
import struct

VERSION=1
KINDS=('W','S','X','XH','K','E','Q','HB','B','SI','PS','MV','WI','F','M')
OPS=('','add','mul','fma','min','max','abs','rsqrt','rcp','rne16','rne16w','shl')
FIELDS=('wload_off','sread_src','gstore_off','xsite_src','xsite_wsrc','shfl_idx','psel','thread_class')
EVENT_KINDS=('ENTER_NODE','EVAL_NODE','MMA','PUBLISH_M','STORE')
DTYPES={1:'|i1',2:'<i2',3:'<i8'}
SPEC_HEADER=struct.Struct('<8sIi32s7IiQQ')
NODE=struct.Struct('<IIiiiiiI4i')
MMA=struct.Struct('<28i')
STORE=struct.Struct('<17i')
CLASS_STORE=struct.Struct('<18i')
ROLE=struct.Struct('<iQ')
EVENT=struct.Struct('<IIii')
MAP_HEADER=struct.Struct('<8sIIQ32s')
MAP_ENTRY=struct.Struct('<III5QQQ32s')
DOMAIN_HEADER=struct.Struct('<8sII32s')
NT=0x7c00-0x410
MAX_MAP_BYTES=512<<20
MAX_INDEX_BYTES=8<<20
MAX_METADATA_BYTES=8<<20
RULES={'version':VERSION,'endian':'little','integer_types':'i32/u32/i64/u64',
       'msvc':'/O2 /fp:strict /fp:except- C++17',
       'gcc_clang':'-O2 -fno-fast-math -ffp-contract=off C++17',
       'primitive_version':'frozen-G-g2-no-arithmetic-edit',
       'backend':'scalar-portable-cpp17-iga-original-worker',
       'kinds':list(KINDS),'ops':list(OPS),'map_fields':list(FIELDS),'event_kinds':list(EVENT_KINDS),
       'event_usage':'IG-a representation evidence only; original recursive Worker executes unchanged'}

def sha(data):return hashlib.sha256(data).hexdigest()

def encode(value):
    """Small metadata/document TLV, bounded lengths, exact insertion order."""
    if value is None:return b'N'
    if type(value) is bool:return b'T' if value else b'F'
    if type(value) is int:return b'I'+struct.pack('<q',value)
    if type(value) is float:return b'R'+struct.pack('<d',value)
    if isinstance(value,(str,bytes)):
        raw=value.encode('utf-8') if isinstance(value,str) else value
        return (b'S' if isinstance(value,str) else b'B')+struct.pack('<Q',len(raw))+raw
    if isinstance(value,(list,tuple)):
        return b'L'+struct.pack('<Q',len(value))+b''.join(encode(x) for x in value)
    if type(value) is dict:
        return b'D'+struct.pack('<Q',len(value))+b''.join(encode(k)+encode(v) for k,v in value.items())
    raise ValueError('unsupported fixed metadata type')

def decode(raw,max_bytes=MAX_INDEX_BYTES):
    if len(raw)>max_bytes:raise ValueError('binary metadata bound')
    pos=0
    def take(n):
        nonlocal pos
        if n<0 or n>len(raw)-pos:raise ValueError('truncated binary metadata')
        part=raw[pos:pos+n];pos+=n;return part
    def item(depth=0):
        if depth>64:raise ValueError('binary metadata depth')
        tag=take(1)
        if tag==b'N':return None
        if tag in (b'T',b'F'):return tag==b'T'
        if tag in (b'I',b'R'):return struct.unpack('<q' if tag==b'I' else '<d',take(8))[0]
        if tag in (b'S',b'B',b'L',b'D'):
            n=struct.unpack('<Q',take(8))[0]
            if n>max_bytes:raise ValueError('binary metadata length')
            if tag in (b'S',b'B'):
                val=take(n);return val.decode('utf-8') if tag==b'S' else val
            if tag==b'L':return [item(depth+1) for _ in range(n)]
            result={}
            for _ in range(n):
                key=item(depth+1)
                if type(key) not in (int,str) or key in result:raise ValueError('duplicate/invalid binary metadata key')
                result[key]=item(depth+1)
            return result
        raise ValueError('unknown binary metadata tag')
    result=item()
    if pos!=len(raw):raise ValueError('trailing binary metadata')
    return result

def normalized(wiring,contract):
    nodes=[]
    for n in wiring['nodes']:
        kind=n[0];op='';a=b=c=d=k=0;args=[]
        if kind=='F':op=n[1];args=n[2]
        elif kind=='WI':op=n[1];a,b,c,k=n[2:6]
        else:
            if len(n)>1:a=n[1]
            if len(n)>2:b=n[2]
            if len(n)>3:c=n[3]
        if kind not in KINDS or op not in OPS or len(args)>4:raise ValueError('node encoding')
        nodes.append((KINDS.index(kind),OPS.index(op),a,b,c,d,k,len(args),*(args+[0]*(4-len(args)))))
    mmas=[]
    for m in wiring['mmas']:
        if tuple(len(m[x]) for x in ('A','B','C'))!=(16,8,4):raise ValueError('bad MMA operand length')
        mmas.append(tuple(m['A']+m['B']+m['C']))
    con=[-1]*len(nodes)
    for key,val in contract.items():
        i=int(key)
        if not 0<=i<len(nodes) or val not in (0,1):raise ValueError('bad contraction')
        con[i]=val
    stores=[];classes=[]
    for pc in wiring['gstore_pcs']:
        atoms=[0]*16 if 'classes' in wiring else wiring['gstore_atoms'][str(pc)]
        if len(atoms)!=16:raise ValueError('store not 16 bytes')
        stores.append(tuple([pc]+atoms))
    for cl in wiring.get('classes',[]):
        for pc in cl['gstore_pcs']:
            atoms=cl['gstore_atoms'][str(pc)]
            if len(atoms)!=16:raise ValueError('class store width')
            classes.append(tuple([cl['cls'],pc]+atoms))
    return {'nodes':nodes,'mmas':mmas,'contract':con,'stores':stores,'class_stores':classes}

def pack_spec(launch,key,wiring,contract,sp):
    n=normalized(wiring,contract)
    counts=[len(n[k]) for k in ('nodes','mmas','stores','class_stores')]+[len(sp[k]) for k in ('inputs','outputs','releases')]
    header=SPEC_HEADER.pack(b'IG1SPEC1',VERSION,launch,bytes.fromhex(key),*counts,sp['weight_word'],sp['weight_bytes'],sp['output_bytes'])
    raw=header+b''.join(NODE.pack(*x) for x in n['nodes'])+b''.join(MMA.pack(*x) for x in n['mmas'])
    raw+=struct.pack('<'+str(len(n['contract']))+'i',*n['contract'])
    raw+=b''.join(STORE.pack(*x) for x in n['stores'])+b''.join(CLASS_STORE.pack(*x) for x in n['class_stores'])
    raw+=b''.join(ROLE.pack(*x) for name in ('inputs','outputs','releases') for x in sp[name])
    if len(raw)>MAX_METADATA_BYTES:raise ValueError('native metadata bound')
    return raw

def first_evaluation_events(wiring,contract):
    """Mirror original Worker::val/mma/run calls, no topological optimizer.

    This table is representation evidence in IG-a. Original Worker still runs.
    ENTER/EVAL preserve parent-before-child marking and child-before-parent
    arithmetic; MMA/PUBLISH and STORE preserve the exact original stage order.
    """
    s=normalized(wiring,contract);nodes=s['nodes'];seen=[0]*len(nodes);events=[]
    matoms=[-1]*(4*len(s['mmas']))
    for i,n in enumerate(nodes):
        if KINDS[n[0]]=='M':
            if not 0<=n[2]<len(s['mmas']) or not 0<=n[3]<4:raise ValueError('bad M atom')
            matoms[4*n[2]+n[3]]=i
    if -1 in matoms:raise ValueError('missing M atom')
    def val(i,stage):
        if not 0<=i<len(nodes):raise IndexError('node id')
        if seen[i]==2:return
        if seen[i]==1:raise RuntimeError('cycle in wiring DAG')
        n=nodes[i];kind=KINDS[n[0]];op=OPS[n[1]];a,b,c=n[2:5];args=list(n[8:8+n[7]])
        if kind=='M':raise RuntimeError('MMA value before its stage')
        seen[i]=1;events.append((0,i,stage,-1))
        deps=[]
        if kind in ('E','Q'):deps=[a]
        elif kind=='HB':deps=[a,b]
        elif kind in ('B','SI'):deps=[b]
        elif kind in ('PS','MV','WI'):deps=[b,c]
        elif kind=='F':
            if op=='add' and s['contract'][i]>=0:
                m0,m1=(nodes[x] for x in args)
                if any(KINDS[m[0]]!='F' or OPS[m[1]]!='mul' or m[8]!=m[9] for m in (m0,m1)):raise ValueError('bad contraction DAG')
                k=s['contract'][i];deps=[nodes[args[k]][8],args[1-k]]
            else:deps=args
        for dep in deps:val(dep,stage)
        events.append((1,i,stage,-1));seen[i]=2
    for stage,m in enumerate(s['mmas']):
        for i in m:val(i,stage)
        events.append((2,stage,stage,-1))
        for j in range(4):
            atom=matoms[4*stage+j];events.append((3,atom,stage,-1));seen[atom]=2
    stores=s['class_stores'] if 'classes' in wiring else s['stores']
    for index,st in enumerate(stores):
        for atom in st[2:] if 'classes' in wiring else st[1:]:val(atom,len(s['mmas']))
        events.append((4,index,len(s['mmas']),st[0] if 'classes' in wiring else -1))
    return b''.join(EVENT.pack(*e) for e in events)

def unpack_spec(raw,expected_key=None,expected_launch=None):
    if len(raw)<SPEC_HEADER.size or len(raw)>MAX_METADATA_BYTES:raise ValueError('truncated/oversized plan metadata')
    h=SPEC_HEADER.unpack_from(raw)
    magic,version,launch,key=h[:4];nn,nm,ns,nc,ni,no,nr=h[4:11];word,weight,out=h[11:]
    if magic!=b'IG1SPEC1' or version!=VERSION:raise ValueError('plan version')
    if expected_key is not None and key.hex()!=expected_key:raise RuntimeError('plan identity mismatch')
    if expected_launch is not None and launch!=expected_launch:raise ValueError('plan launch')
    size=SPEC_HEADER.size+nn*(NODE.size+4)+nm*MMA.size+ns*STORE.size+nc*CLASS_STORE.size+(ni+no+nr)*ROLE.size
    if size!=len(raw):raise ValueError('truncated/extra plan payload')
    pos=SPEC_HEADER.size
    def records(fmt,count):
        nonlocal pos
        vals=[fmt.unpack_from(raw,pos+i*fmt.size) for i in range(count)];pos+=count*fmt.size;return vals
    nodes=records(NODE,nn);mmas=records(MMA,nm);con=[x[0] for x in records(struct.Struct('<i'),nn)]
    for node in nodes:
        if node[0]>=len(KINDS) or node[1]>=len(OPS) or node[7]>4:raise ValueError('plan node opcode/arity')
    if any(v not in (-1,0,1) for v in con):raise ValueError('bad contraction')
    stores=records(STORE,ns);classes=records(CLASS_STORE,nc)
    return {'launch':launch,'identity':key.hex(),'nodes':nodes,'mmas':mmas,'contract':con,
            'stores':stores,'class_stores':classes,'inputs':records(ROLE,ni),'outputs':records(ROLE,no),'releases':records(ROLE,nr),
            'weight_word':word,'weight_bytes':weight,'output_bytes':out}
