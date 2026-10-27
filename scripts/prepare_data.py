"""Recover numeric arrays from the original release without unpickling objects."""
import argparse
from pathlib import Path
import hashlib
import json
import tempfile
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
COMMIT='6020df5802e36a7786fbc1091efcb37b704bea41'
BASE=f'https://raw.githubusercontent.com/LeaCasse/QRU_Calorimetry_Optimization/{COMMIT}/Dataset/'


def loader_arrays(path):
    # This parser is specific to the two released float32 TensorDatasets.
    # No pickle opcode is interpreted and no Python object is instantiated.
    with zipfile.ZipFile(path) as z:
        root=z.namelist()[0].split('/')[0]
        if z.read(root+'/byteorder')!=b'little':raise ValueError('Unexpected byte order')
        x=np.frombuffer(z.read(root+'/data/0'),dtype='<f4').copy()
        y=np.frombuffer(z.read(root+'/data/1'),dtype='<f4').copy()
    if len(x)!=3*len(y) or not set(np.unique(y)).issubset({-1.,0.,1.}):
        raise ValueError('Unexpected tensor shape or target codes')
    return x.reshape(-1,3),y


def prepare(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    workbook=pd.read_excel(source/'dataset.xlsx').apply(pd.to_numeric,errors='coerce').dropna()
    raw=workbook.iloc[:,:3].values.astype(float)
    norm=(raw-raw.min(0))/(raw.max(0)-raw.min(0))*2*np.pi-np.pi
    tree=cKDTree(norm);arrays={};frames=[];maximum=0.
    for split in ['train','test']:
        x,y=loader_arrays(source/f'qml_essai_{split}_loader.pth')
        distance,index=tree.query(x);maximum=max(maximum,float(distance.max()))
        if distance.max()>=1e-5 or not np.array_equal(y,workbook.label.values[index]):
            raise ValueError('Loader and spreadsheet do not match')
        arrays['x_'+split]=x;arrays['y_'+split]=y;arrays['raw_'+split]=raw[index]
        frame=pd.DataFrame(raw[index],columns=workbook.columns[:3]);frame['label']=y.astype(int)
        frame['partition']=split;frame['source_row']=workbook.index.values[index]+2;frames.append(frame)
    np.savez_compressed(out/'calorimetry.npz',**arrays)
    pd.concat(frames).to_csv(out/'events.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    provenance={'source_commit':COMMIT,'source_base_url':BASE,'features':list(workbook.columns[:3]),
        'feature_semantics':'Stored headers only. Physical definitions and units are unverified.',
        'labels':[-1,0,1],'particle_label_mapping':'Unverified in supplied files',
        'loader_train_count':len(arrays['y_train']),'loader_test_count':len(arrays['y_test']),
        'spreadsheet_numeric_rows':len(workbook),'max_reconstruction_distance':maximum,
        'source_file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir');parser.add_argument('--download',action='store_true')
    parser.add_argument('--output',default=str(ROOT/'data'));args=parser.parse_args()
    if args.download:
        hashes=json.loads((ROOT/'data/provenance.json').read_text())['source_file_sha256']
        with tempfile.TemporaryDirectory() as folder:
            for name,expected in hashes.items():
                path=Path(folder)/name;urllib.request.urlretrieve(BASE+name,path)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Source checksum mismatch')
            prepare(folder,args.output)
    elif args.source_dir:prepare(args.source_dir,args.output)
    else:parser.error('Use --source-dir or --download')
