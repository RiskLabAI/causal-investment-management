"""Preserve the original primitive draws for the public causal-role examples."""

import json
import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value



def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter01_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed,count,columns in [(2301,20000,9),(2302,50000,10),(2303,8,2)]:
        if seed==2303:
            values=np.random.RandomState(seed).rand(count,columns)
        else:
            generator=np.random.default_rng(seed)
            values=np.column_stack([generator.normal(size=count) for _ in range(columns)])
        path=destination/f'primitive_{seed}.csv'
        save_exact(path,pd.DataFrame(values,columns=[f'z{i+1}' for i in range(columns)]).to_csv(index=False).encode())
        records.append(dict(path=str(path),name=path.name,sha256=digest(path),seed=seed,rows=count,columns=columns))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),primitive='Original NumPy normal innovations and unfitted uniform graph-layout initial positions. Julia computes all generated observations, fits, graph decisions, force-directed coordinates, summaries and figures.')
    (REPORT/'cim_chapter01_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),primitive_values=sum(r['rows']*r['columns'] for r in records)),indent=2))


if __name__=='__main__':
    main()
