"""Preserve the original unfitted random inputs for the Chapter 3 experiments."""

import json
from collections import OrderedDict

import numpy as np

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT = '03_structural_treatment_effects'
STEMS = ['03_01_randomized_contrasts_adjustment_average_effects','03_02_difference_in_differences_instrumental_variables','03_03_structural_causal_models_deterministic_evaluation']
DIRECTORY = ROOT/'notebooks'/'julia'/'data'/'chapter03_random_inputs'


def blocks(section):
    if section == 'average':
        rng = np.random.default_rng(250101)
        covariate = rng.normal(size=20000)
        shock = rng.standard_normal(20000)
        random_assignment = rng.binomial(1, .5, 20000)
        selected_probability = 1/(1+np.exp(-(-.2+1.1*covariate)))
        selected_assignment = rng.binomial(1, selected_probability)
        yield 'population',250101,np.column_stack([covariate,shock,random_assignment,selected_assignment])
        for repetition in range(500):
            seed = 251000+repetition
            rng = np.random.default_rng(seed)
            yield 'recovery',seed,np.column_stack([rng.integers(0,20000,600),rng.binomial(1,.5,600)])
        for count in [50,100,250,500,1000,2500]:
            for repetition in range(350):
                seed = 252000+1000*count+repetition
                rng = np.random.default_rng(seed)
                yield 'size_'+str(count),seed,np.column_stack([rng.integers(0,20000,count),rng.binomial(1,.5,count)])
    elif section == 'did_iv':
        for index in range(4):
            seed = 253000+index
            rng = np.random.default_rng(seed)
            yield 'did_demo',seed,np.column_stack([rng.standard_normal(8000) for _ in range(3)])
        for index in range(3):
            for repetition in range(500):
                seed = 254000+index*1000+repetition
                rng = np.random.default_rng(seed)
                yield 'did_'+str(index),seed,np.column_stack([rng.standard_normal(500) for _ in range(3)])
        for index in range(3):
            for repetition in range(500):
                seed = 255000+index*1000+repetition
                rng = np.random.default_rng(seed)
                yield 'iv_'+str(index),seed,np.column_stack([rng.standard_normal(1200) for _ in range(4)])
    elif section == 'scm':
        rng = np.random.default_rng(256001)
        yield 'chain',256001,np.column_stack([rng.standard_normal(500) for _ in range(3)])
        yield 'nonlinear',256001,np.column_stack([rng.standard_normal(700) for _ in range(2)])
        yield 'fork',256001,np.column_stack([rng.standard_normal(1000) for _ in range(3)])
    else:
        raise ValueError(section)


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    files = []
    for section in ['average','did_iv','scm']:
        path = DIRECTORY/(section+'_primitives.f64le')
        exists = path.exists()
        groups = OrderedDict()
        offset = 0
        with path.open('rb' if exists else 'xb') as stream:
            for name,seed,values in blocks(section):
                content = values.astype('<f8',copy=False).tobytes(order='F')
                if exists:
                    assert stream.read(len(content))==content,(section,name,seed)
                else:
                    stream.write(content)
                group = groups.setdefault(name,dict(seeds=[],rows=[],columns=values.shape[1],offset_bytes=[]))
                assert group['columns']==values.shape[1]
                group['seeds'].append(seed)
                group['rows'].append(len(values))
                group['offset_bytes'].append(offset)
                offset += len(content)
            assert not exists or stream.read(1)==b''
        manifest = DIRECTORY/(section+'_inputs.toml')
        content = '[file]\nname = '+json.dumps(path.name)+'\nsha256 = '+json.dumps(digest(path))+'\n\n'
        for name,group in groups.items():
            content += '[groups.'+json.dumps(name)+']\n'+''.join(key+' = '+json.dumps(value)+'\n' for key,value in group.items())+'\n'
        save_exact(manifest,content.encode())
        files.extend([dict(path=str(path),sha256=digest(path),primitive_values=offset//8),dict(path=str(manifest),sha256=digest(manifest))])
    data_path = ROOT/'data'/'public'/'callaway_santanna'/'mpdta.csv'
    assert digest(data_path)=='1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661'
    result = dict(files=files,public_data=dict(path=str(data_path),sha256=digest(data_path)),primitive_values=sum(row.get('primitive_values',0) for row in files),description='Original standard-normal draws, resampling indices, and treatment assignments. Conditional binomial draws retain the original covariate-defined assignment probabilities. Original seeds and generator call order are unchanged. Julia generates potential outcomes, structural observations, estimates and diagnostics. No fitted result is stored as an input.')
    save_exact(REPORT/'cim_chapter03_random_input_preparation.json',(json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(files=len(files),primitive_values=result['primitive_values'],public_data_unchanged=True),indent=2))


if __name__=='__main__':
    main()
