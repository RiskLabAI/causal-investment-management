"""Store exact primitive normal streams for the original identification experiments."""

import json
import sys
import numpy as np

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value


def plans(section):
    if section=='backdoor':
        cases=[(240101,50000)]+[(240100+n,n) for n in [250,500,1000,2500,5000,10000,50000]]
        cases += [(240200+r+int(delta*1000),1500) for delta in [0.,.4,.8,1.2] for r in range(150)]
        return [(f'backdoor_{seed}_{count}',seed,[count],6) for seed,count in cases]
    if section=='frontdoor':
        return [('fd_demo',20260706,[6000]*2,5),('fd_recovery',102000001,[2000]*200,5),('fd_convergence',102000002,[2000]*400,5),('fd_wedge',102000004,[2000]*600,5),('fd_coverage',102000003,[2000]*200,5),('fd_intervals',240202,[40000],5),('fd_direct',3,[20000]*2,5),('fd_large',1,[60000],5)]
    if section=='iv':
        return [('iv_demo',20260707,[3000]*3,5),('iv_recovery',80808081,[2000]*400,5),('iv_wedge',80808082,[2000]*1200,5),('iv_ar',80808083,[2000]*1600,5),('iv_limiting',80808084,[2000]*800,5),('iv_sets',3,[4000,300],5)]
    raise ValueError(section)


def main(section=None):
    section=section or sys.argv[1]
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter02_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/(section+'_normals.f64le')
    plan=plans(section)
    groups=[]
    offset=0
    exists=path.exists()
    with path.open('rb' if exists else 'xb') as stream:
        for name,seed,counts,columns in plan:
            generator=np.random.default_rng(seed)
            offsets=[]
            for count in counts:
                offsets.append(offset)
                values=np.column_stack([generator.standard_normal(count) for _ in range(columns)]).astype('<f8',copy=False)
                content=values.tobytes(order='F')
                if exists:
                    assert stream.read(len(content))==content,(section,name,count)
                else:
                    stream.write(content)
                offset+=len(content)
            groups.append(dict(name=name,seed=seed,rows=counts,columns=columns,offset_bytes=offsets))
        if exists:
            assert stream.read(1)==b''
    record=dict(path=str(path),name=path.name,sha256=digest(path),bytes=offset,primitive_values=offset//8)
    manifest=destination/(section+'_inputs.toml')
    manifest_text='[file]\nname = '+json.dumps(path.name)+'\nsha256 = '+json.dumps(record['sha256'])+'\n\n'
    for group in groups:
        manifest_text+='[groups.'+json.dumps(group['name'])+']\n'
        for key in ['seed','rows','columns','offset_bytes']:
            manifest_text+=key+' = '+json.dumps(group[key])+'\n'
        manifest_text+='\n'
    files=[record]
    if section=='iv':
        generator=np.random.default_rng(240303)
        uniform=generator.random(100000)
        assignment=generator.integers(0,2,size=100000)
        baseline=generator.standard_normal(100000)
        compliance=destination/'iv_compliance.f64le'
        save_exact(compliance,np.column_stack([uniform,assignment,baseline]).astype('<f8',copy=False).tobytes(order='F'))
        files.append(dict(path=str(compliance),name=compliance.name,sha256=digest(compliance),bytes=compliance.stat().st_size,primitive_values=300000))
        manifest_text+='[compliance]\nname = '+json.dumps(compliance.name)+'\nsha256 = '+json.dumps(files[-1]['sha256'])+'\nrows = 100000\ncolumns = 3\n'
    save_exact(manifest,manifest_text.encode())
    result=dict(files=files,manifest=dict(path=str(manifest),sha256=digest(manifest)),groups=groups,primitive='Original NumPy normal innovations in column-major little-endian Float64 format. Each block preserves the original seed and generator call order. Binary storage avoids decimal serialization overhead. Julia computes all structural observations and estimators. The IV compliance inputs contain unfitted uniform draws, randomized binary assignments and normal baselines.')
    (REPORT/('cim_chapter02_'+section+'_random_input_preparation.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(section=section,groups=len(groups),blocks=sum(len(group['rows']) for group in groups),primitive_values=sum(item['primitive_values'] for item in files)),indent=2))


if __name__=='__main__':
    main()
