"""Build every revised figure and table from saved numerical results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse
import gzip
import itertools
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.metrics import confusion_matrix
from qru.data import ROOT,load_data
from qru.models import QRU,classify

FIG=ROOT/'results/figures';TAB=ROOT/'results/tables'
COLORS=['#2463a4','#de7b2c','#25816b','#9a5b96','#6c6f7d','#b29b25','#a34747']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.prop_cycle':plt.cycler(color=COLORS),'savefig.bbox':'tight','pdf.fonttype':42,
    'axes.labelsize':11,'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10})


def runs(stage):
    paths=list((ROOT/'results'/stage).glob('*.json'))
    rows={}
    archive=ROOT/'results/runs.json.gz'
    if archive.exists():
        with gzip.open(archive,'rt') as f:rows={r['id']:r for r in json.load(f) if r['config']['stage']==stage}
    for p in paths:
        r=json.loads(p.read_text());rows[r['id']]=r
    return list(rows.values())


def model_arrays(stage,run_id):
    path=ROOT/'results'/stage/(run_id+'.npz')
    if path.exists():return dict(np.load(path))
    with np.load(ROOT/'results/models.npz') as a:
        prefix=run_id+'_'
        return {key[len(prefix):]:a[key] for key in a.files if key.startswith(prefix)}


def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=160);plt.close(fig)


def table(name,caption,label,columns,rows,align=None):
    align=align or 'l'+'r'*(len(columns)-1)
    text='\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text+='\\small\n\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(columns)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (TAB/(name+'.tex')).write_text(text)


def stat(rows,section,key):
    a=np.array([r[section][key] for r in rows]);return a.mean(),a.std(ddof=1) if len(a)>1 else 0


def mean_sd(a,percent=False):
    a=np.asarray(a);factor=100 if percent else 1
    return f'${factor*a.mean():.2f} \\pm {factor*a.std(ddof=1):.2f}$' if len(a)>1 else f'{factor*a.mean():.2f}'


def main(overleaf=None):
    FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
    d=load_data();sw=runs('sweeps');comp=runs('comparison');features=runs('feature_ablation')
    classical=json.loads((ROOT/'results/classical.json').read_text())
    summary=json.loads((ROOT/'results/search_summary.json').read_text())
    stats={}
    # Spectral support under fixed encoding factors.
    s=pd.read_csv(ROOT/'results/spectrum.csv');fig,ax=plt.subplots(figsize=(7,2.8))
    ax.bar(s.order-.17,s.QRU_fixed_activation,.34,label='QRU, fixed factors')
    ax.bar(s.order+.17,s.VQC_activation,.34,label='VQC')
    ax.set(xlabel='Coordinate Fourier order',ylabel='Activation rate',xticks=range(-8,9),ylim=(0,1.02));ax.legend(frameon=False)
    save(fig,'spectrum')
    # Released raw columns, without assigning unverified physical meanings.
    fig,axs=plt.subplots(1,3,figsize=(10,2.7))
    for j,name in enumerate(['meanz','shower_length','energ_ecal']):
        for k,label in enumerate([-1,0,1]):
            axs[j].hist(d['raw_train'][d['y_train']==label,j],bins=40,density=True,histtype='step',lw=1.5,label=f'Code {label}')
        axs[j].set(xlabel=name,ylabel='Density')
    axs[0].legend(frameon=False);save(fig,'features')
    # Depth trajectories use validation only.
    dep=sorted({r['config']['depth'] for r in sw if r['config']['group']=='depth'})
    fig,axs=plt.subplots(1,2,figsize=(8,3))
    for z in dep:
        rr=[r for r in sw if r['config']['group']=='depth' and r['config']['depth']==z]
        for ax,key in zip(axs,['val_mse','val_accuracy']):
            v=np.array([[t[key] for t in r['fit']['trace']] for r in rr]);t=np.arange(v.shape[1])
            ax.plot(t,v.mean(0),label=str(z));ax.fill_between(t,v.mean(0)-v.std(0),v.mean(0)+v.std(0),alpha=.08)
    axs[0].set(xlabel='Epoch',ylabel='Validation MSE');axs[0].set_yscale('log')
    axs[1].set(xlabel='Epoch',ylabel='Validation accuracy');axs[1].legend(title='Depth',frameon=False,ncol=2)
    save(fig,'depth_curves')
    rows=[]
    for z in dep:
        rr=[r for r in sw if r['config']['group']=='depth' and r['config']['depth']==z]
        rows.append([z,9*z,mean_sd([r['validation']['accuracy'] for r in rr],True),f'{stat(rr,"validation","mse")[0]:.4f}'])
    table('depth','Depth sweep','tab:depth',['$L$','$P$','Validation accuracy (\\%)','MSE'],rows)
    # Small sweep panels, all on the same external validation metrics.
    specs=[('normalization','normalization',['centered','positive','standard']),('axes','axes',['XYX','XZX','YXY','YZY','ZXZ','ZYZ','XYZ']),
           ('parameters','parameters',[1,2,3,4,5]),('optimizer','optimizer',['SGD','RMSprop','Adam','Adagrad','Adadelta']),
           ('loss','loss',['L1','L2','Huber']),('learning_rate','lr',[5e-7,5e-5,5e-4,.003,.03,.5])]
    for group,key,vals in specs:
        fig,axs=plt.subplots(1,2,figsize=(8,2.8));points=[]
        for value in vals:
            rr=[r for r in sw if r['config']['group']==group and r['config'].get(key)==value]
            points.append([*stat(rr,'validation','accuracy'),*stat(rr,'validation','mse')])
        a=np.asarray(points);t=np.arange(len(vals))
        labels=[f'{v:g}' if isinstance(v,float) else str(v) for v in vals]
        for ax,col,ylabel in [(axs[0],0,'Validation accuracy'),(axs[1],2,'Validation MSE')]:
            ax.errorbar(t,a[:,col],yerr=a[:,col+1],fmt='o',capsize=3,color=COLORS[0]);ax.set_xticks(t,labels,rotation=25 if group in ['optimizer','normalization','learning_rate'] else 0)
            ax.set_ylabel(ylabel)
        save(fig,group)
    # Nine function draws, shown as illustrations rather than an expressivity proof.
    u=np.linspace(-np.pi,np.pi,401);fig,axs=plt.subplots(2,1,figsize=(7,5),sharex=True,sharey=True)
    rng=np.random.default_rng(2026);params=rng.uniform(-np.pi,np.pi,(9,12))
    for ax,axes in zip(axs,['XYX','XYZ']):
        m=QRU(4,dimensions=1,axes=axes)
        for p in params:ax.plot(u,m.forward(p,u[:,None]),lw=1)
        ax.set(ylabel=r'$\langle Z\rangle$',title=axes,ylim=(-1.05,1.05))
    axs[-1].set_xlabel('Input angle');save(fig,'templates')
    # Accumulation is evaluated against both epochs and optimizer steps.
    fig,axs=plt.subplots(1,2,figsize=(8,3));batch_rows=[]
    for b in [1,5,20,100,256,'full']:
        rr=[r for r in sw if r['config']['group']=='batch' and r['config']['batch_size']==b]
        a=np.array([[t['val_accuracy'] for t in r['fit']['trace']] for r in rr]);t=np.arange(a.shape[1]);steps=rr[0]['fit']['steps']/10
        axs[0].plot(t,a.mean(0),label=str(b));axs[1].plot(t*steps,a.mean(0),label=str(b))
        batch_rows.append([str(b),rr[0]['fit']['steps'],mean_sd([r['validation']['accuracy'] for r in rr],True)])
    axs[0].set(xlabel='Epoch',ylabel='Validation accuracy');axs[1].set(xlabel='Optimizer steps',ylabel='Validation accuracy',xscale='symlog')
    axs[0].legend(title='Batch size',frameon=False,ncol=2);save(fig,'batch')
    table('batch','Update budget','tab:batch',['Batch size','Updates','Validation accuracy (\\%)'],batch_rows)
    # Fixed schedules are not adaptive to the observed validation trace.
    fig,axs=plt.subplots(1,2,figsize=(8,2.8))
    for opt in ['Adam','SGD']:
        for schedule in [False,True]:
            rr=[r for r in sw if r['config']['group']=='schedule' and r['config']['optimizer']==opt and r['config']['schedule']==schedule]
            values=np.array([[t['val_accuracy'] for t in r['fit']['trace']] for r in rr]);t=np.arange(values.shape[1])
            axs[0].plot(t,values.mean(0),label=opt+(' schedule' if schedule else ' fixed'))
            c=rr[0]['config'];axs[1].step(t,c['lr']*np.where(t>=25,.1,1)*np.where(t>=37,.1,1) if schedule else np.full(len(t),c['lr']),where='post')
    axs[0].set(xlabel='Epoch',ylabel='Validation accuracy');axs[0].legend(frameon=False)
    axs[1].set(xlabel='Epoch',ylabel='Learning rate',yscale='log');save(fig,'schedule')
    # Search results and their real budgets.
    gp=json.loads((ROOT/'results/bayesian_search.json').read_text());hb=json.loads((ROOT/'results/hyperband_search.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(8,2.8));scores=[r['result']['validation']['mse'] for r in gp]
    axs[0].plot(range(1,len(scores)+1),scores,'o',ms=3,alpha=.5,label='Trial');axs[0].plot(range(1,len(scores)+1),np.minimum.accumulate(scores),label='Best so far')
    axs[0].set(xlabel='GP trial',ylabel='Validation MSE',yscale='log');axs[0].legend(frameon=False)
    for budget in [1,3,9,27]:
        rr=[r for r in hb if r['result']['config']['epochs']==budget]
        axs[1].scatter([budget]*len(rr),[r['result']['validation']['mse'] for r in rr],alpha=.7,s=16)
    axs[1].set(xlabel='Hyperband epochs',ylabel='Validation MSE',yscale='log',xscale='log',xticks=[1,3,9,27],xticklabels=[1,3,9,27]);save(fig,'search')
    search_rows=[]
    for key,label in [('bayesian_best_mse','GP, MSE'),('bayesian_best_accuracy','GP, accuracy'),('hyperband_best_full_budget','Hyperband')]:
        r=summary[key]['result'];c=r['config']
        search_rows.append([label,c['depth'],f"{c['lr']:g}",c['loss'],c['optimizer'],f"{r['validation']['mse']:.4f}",f"{100*r['validation']['accuracy']:.2f}"])
    table('search','Search results','tab:search',['Selection','$L$','$\\eta$','Loss','Optimizer','MSE','Acc. (\\%)'],search_rows, 'lrrllrr')
    # Main model comparison and uncertainty across initializations.
    comp_rows=[];fig,axs=plt.subplots(1,2,figsize=(8,3));flat=[]
    grouped=[]
    for kind,depth,label in [('QRU',4,'QRU, L=4'),('QRU',10,'QRU, L=10'),('VQC',10,'VQC, L=10'),('MLP',10,'MLP, scalar')]:
        rr=sorted([r for r in comp if r['config']['kind']==kind and r['config']['depth']==depth],key=lambda r:r['config']['seed'])
        grouped.append((label,rr))
        comp_rows.append([label,rr[0]['fit']['parameter_count'],mean_sd([r['test']['accuracy'] for r in rr],True),f"{stat(rr,'test','mse')[0]:.4f}"])
        stats[kind+str(depth)]={'accuracy':stat(rr,'test','accuracy')[0],'sd':stat(rr,'test','accuracy')[1],'mse':stat(rr,'test','mse')[0]}
        for ax,key in zip(axs,['val_mse','val_accuracy']):
            a=np.array([[t[key] for t in r['fit']['trace']] for r in rr]);t=np.arange(a.shape[1]);ax.plot(t,a.mean(0),label=label);ax.fill_between(t,a.mean(0)-a.std(0),a.mean(0)+a.std(0),alpha=.12)
        for r in rr:flat.append({'model':label,'seed':r['config']['seed'],**r['test'],'validation_mse':r['validation']['mse'],'lr':r['config']['lr']})
    for name,label in [('MLP_softmax','MLP, softmax'),('Logistic','Logistic'),('Fourier','Fourier logits'),('Trees','Boosted trees')]:
        rr=[r for r in classical if r['method']==name]
        comp_rows.append([label,rr[0]['parameter_count'] or 'tree ensemble',mean_sd([r['test_accuracy'] for r in rr],True),f"{np.mean([r['test_mse'] for r in rr]):.4f}"])
    table('comparison','Classifier comparison','tab:comparison',['Model','$P$','Test accuracy (\\%)','MSE'],comp_rows)
    pd.DataFrame(flat).to_csv(ROOT/'results/comparison.csv',index=False)
    axs[0].set(xlabel='Epoch',ylabel='Validation MSE',yscale='log');axs[1].set(xlabel='Epoch',ylabel='Validation accuracy');axs[1].legend(frameon=False,loc='lower right');save(fig,'comparison')
    # Class codes and feature ablation avoid unverified particle assignments.
    fig,axs=plt.subplots(1,2,figsize=(8,3));prediction_bundle={}
    for ax,(label,rr) in zip(axs,[grouped[1],grouped[3]]):
        matrices=[]
        for r in rr:
            ar=model_arrays('comparison',r['id']);pred=ar['prediction'];matrices.append(confusion_matrix(d['y_test'],classify(pred),labels=[-1,0,1],normalize='true'))
            prediction_bundle[r['id']+'_theta']=ar['theta'];prediction_bundle[r['id']+'_prediction']=pred
        cm=np.mean(matrices,axis=0);ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{100*cm[i,j]:.1f}',ha='center',va='center',color='white' if cm[i,j]>.5 else 'black')
        ax.set(title=label,xlabel='Predicted code',ylabel='True code',xticks=range(3),xticklabels=[-1,0,1],yticks=range(3),yticklabels=[-1,0,1])
    save(fig,'confusion')
    frows=[]
    for k in [1,2,3]:
        for choice in itertools.combinations(range(3),k):
            rr=[r for r in features if r['config']['features']==list(choice)]
            frows.append([' + '.join(str(i+1) for i in choice),mean_sd([r['test']['accuracy'] for r in rr],True),f"{stat(rr,'test','mse')[0]:.4f}"])
    table('feature_ablation','Feature ablation','tab:feature',['Input columns','Test accuracy (\\%)','MSE'],frows)
    # Use seed zero, fixed in advance, for the shot diagnostic.
    qr=grouped[1][1][0];pred=model_arrays('comparison',qr['id'])['prediction'];prob=np.clip((pred+1)/2,0,1)
    shot_rows=[];rng=np.random.default_rng(2026)
    for shots in [50,200,1000]:
        low=int(np.ceil((1-.33)*shots/2)-1);high=int(np.floor((1+.33)*shots/2))
        correct=np.where(d['y_test']==-1,binom.cdf(low,shots,prob),np.where(d['y_test']==1,binom.sf(high,shots,prob),binom.cdf(high,shots,prob)-binom.cdf(low,shots,prob)))
        acc=[]
        for _ in range(200):acc.append(np.mean(classify(2*rng.binomial(shots,prob)/shots-1)==d['y_test']))
        shot_rows.append({'shots':shots,'expected_accuracy':float(correct.mean()),'mc_mean':float(np.mean(acc)),'mc_sd':float(np.std(acc,ddof=1))})
    (ROOT/'results/shots.json').write_text(json.dumps({'run_id':qr['id'],'exact_accuracy':float(np.mean(classify(pred)==d['y_test'])),'model':'ideal independent measurement counts, no device noise','repetitions':200,'rows':shot_rows},indent=2))
    fig,axs=plt.subplots(1,2,figsize=(8,2.8));axs[0].errorbar([r['shots'] for r in shot_rows],[r['mc_mean'] for r in shot_rows],yerr=[r['mc_sd'] for r in shot_rows],fmt='o-',capsize=3)
    axs[0].axhline(np.mean(classify(pred)==d['y_test']),color='gray',ls=':',label='Exact');axs[0].set(xlabel='Shots',ylabel='Accuracy',xscale='log');axs[0].legend(frameon=False)
    margin=np.minimum(abs(pred-.33),abs(pred+.33));axs[1].hist(margin,bins=35,color=COLORS[0]);axs[1].set(xlabel='Distance to nearest threshold',ylabel='Events');save(fig,'shots')
    # Runtime uses separate serial measurements.
    runtime=json.loads((ROOT/'results/runtime.json').read_text());rt=[]
    for name in ['QRU4','QRU10','VQC10','MLP10']:
        rr=[r for r in runtime['runs'] if r['model']==name]
        assert len(rr)==3,'Runtime benchmark incomplete'
        rt.append([name,f"{np.median([r['train_seconds'] for r in rr]):.3f}",f"{1000*np.median([r['inference_seconds'] for r in rr]):.3f}"])
    table('runtime','Serial runtime','tab:runtime',['Model','Training (s)','2926 predictions (ms)'],rt)
    stats['classical']={name:float(np.mean([r['test_accuracy'] for r in classical if r['method']==name])) for name in ['Logistic','Fourier','Trees','MLP_softmax']}
    (ROOT/'results/summary.json').write_text(json.dumps(stats,indent=2))
    macros=[]
    for key,values in [('QRUFour',stats['QRU4']),('QRUTen',stats['QRU10']),('VQC',stats['VQC10']),('MLP',stats['MLP10'])]:
        macros.append(f'\\newcommand{{\\{key}Acc}}{{{100*values["accuracy"]:.2f}}}')
        macros.append(f'\\newcommand{{\\{key}Sd}}{{{100*values["sd"]:.2f}}}')
    for key,name in [('Fourier','Fourier'),('Trees','Trees'),('Logistic','Logistic'),('Softmax','MLP_softmax')]:
        macros.append(f'\\newcommand{{\\{key}Acc}}{{{100*stats["classical"][name]:.2f}}}')
    (TAB/'numbers.tex').write_text('\n'.join(macros)+'\n')
    # One compact numerical archive replaces hundreds of cache files in releases.
    all_rows=[];weights={}
    if (ROOT/'results/models.npz').exists():
        with np.load(ROOT/'results/models.npz') as old:weights={key:old[key] for key in old.files if not key.endswith('_validation_prediction')}
    for stage in ['tuning','sweeps','comparison','feature_ablation','bayesian','hyperband']:
        rr=runs(stage);all_rows.extend(rr)
        for r in rr:
            f=ROOT/'results'/stage/(r['id']+'.npz')
            if f.exists():
                a=np.load(f)
                for key in a.files:
                    if key in ['theta','prediction']:weights[r['id']+'_'+key]=a[key]
    archive_tmp=ROOT/'results'/f'runs.{os.getpid()}.tmp.gz'
    with gzip.open(archive_tmp,'wt') as f:json.dump(all_rows,f)
    archive_tmp.replace(ROOT/'results/runs.json.gz')
    weights_tmp=ROOT/'results'/f'models.{os.getpid()}.tmp.npz'
    np.savez_compressed(weights_tmp,**weights)
    weights_tmp.replace(ROOT/'results/models.npz')
    if overleaf:
        out=Path(overleaf);(out/'Images').mkdir(exist_ok=True);(out/'tables').mkdir(exist_ok=True)
        for p in FIG.glob('*.pdf'):shutil.copy2(p,out/'Images'/p.name)
        for p in TAB.glob('*.tex'):shutil.copy2(p,out/'tables'/p.name)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--overleaf');args=parser.parse_args();main(args.overleaf)
