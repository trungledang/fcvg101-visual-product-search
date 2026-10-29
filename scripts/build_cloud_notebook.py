"""Bundle the reviewed source into one portable Kaggle/Colab notebook."""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as z:
        for pattern in ('vps/*.py', 'scripts/pilot.py', 'scripts/inspect_inputs.py', 'tests/*.py'):
            for p in sorted(ROOT.glob(pattern)):
                z.write(p, p.relative_to(ROOT).as_posix())
    code64 = base64.b64encode(payload.getvalue()).decode()
    def md(s):
        return {'cell_type':'markdown','id':hashlib.sha256(s.encode()).hexdigest()[:12],
                'metadata':{},'source':s.splitlines(True)}
    def code(s):
        return {'cell_type':'code','id':hashlib.sha256(s.encode()).hexdigest()[:12],
                'metadata':{},'source':s.splitlines(True),'outputs':[], 'execution_count':None}
    cells = [md('''# DeepFashion In-shop — frozen OpenAI CLIP ViT-B/32

Feasibility experiment for FCVG101. Full official query/gallery partitions, exact cosine retrieval,
Recall@1/5 and deterministic diagnostic examples. No training or tuning on test results.
Private educational use. No dataset images or weights are embedded in this notebook.

Kaggle: attach your private In-shop files, select a free GPU and enable Internet.
Colab: select a GPU, mount Drive and set `INPUT_ROOT` below to your existing dataset directory.
The runtime's PyTorch is preserved; all other experimental dependencies are pinned.
'''), code('''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6',
    'ftfy==6.3.1', 'regex==2026.9.29', 'ImageHash==4.3.2',
    'matplotlib==3.11.2', 'pytest==9.1.1'], check=True)
'''), code('''from pathlib import Path
import base64, io, zipfile, json, os, shutil, time
import torch
assert torch.cuda.is_available(), 'GPU unavailable: select a free GPU runtime before running.'
WORK = Path('/kaggle/working' if Path('/kaggle/working').is_dir() else '/content') / 'inshop-baseline'
WORK.mkdir(parents=True, exist_ok=True)
INPUT_ROOT = Path('/kaggle/input')  # Colab: replace with your mounted Drive directory
DATA = WORK / 'data'
DATA.mkdir(exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
torch.manual_seed(0)
print('GPU:', torch.cuda.get_device_name(0))
'''), md('## Reviewed implementation\nThe compressed payload contains the project Python source and correctness tests only.\n'), code(f"zipfile.ZipFile(io.BytesIO(base64.b64decode({code64!r}))).extractall(WORK)\n"), code('''# Resolve files without assuming Kaggle's dataset mount naming convention.
input_files, image_dirs = {}, []
for parent, dirs, files in os.walk(INPUT_ROOT):
    p = Path(parent)
    for name in files:
        input_files.setdefault(name, []).append(p / name)
    if (p / 'MEN').is_dir() and (p / 'WOMEN').is_dir():
        image_dirs.append(p)
        dirs[:] = []
    if len(p.relative_to(INPUT_ROOT).parts) >= 7:
        dirs[:] = []
def one(name):
    matches = input_files.get(name, [])
    assert len(matches) == 1, f'Expected exactly one {name}; found {matches}'
    return matches[0]
partition_file = one('list_eval_partition.txt')
bbox_file = one('list_bbox_inshop.txt')
from vps.annotations import file_sha256
# Checksums of the actual user-provided files inspected locally, not publisher hashes.
assert file_sha256(partition_file) == 'cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c'
assert file_sha256(bbox_file) == 'b1a67bf2b1bb22ce57b63238dfabfd987f21ef85e93448f01735148f5848f440'
(DATA / 'Eval').mkdir(exist_ok=True)
shutil.copyfile(partition_file, DATA / 'Eval/list_eval_partition.txt')
# .bin preserves the original ZIP bytes, avoiding remote auto-extraction.
if 'img_archive.bin' in input_files or 'img.zip' in input_files:
    archive = one('img_archive.bin' if 'img_archive.bin' in input_files else 'img.zip')
    assert file_sha256(archive) == '57b4974d629bfd5e539381ccd339d281706dc4369cb370e0b0a1a63a2f8c4dde'
    with zipfile.ZipFile(archive) as z:
        assert all(not x.startswith('/') and '..' not in Path(x).parts for x in z.namelist())
        z.extractall(DATA)
elif len(image_dirs) == 1:
    if not (DATA / 'img').exists():
        (DATA / 'img').symlink_to(image_dirs[0], target_is_directory=True)
else:
    raise FileNotFoundError('Attach the private In-shop archive or extracted img directory')
sys.path.insert(0, str(WORK))
from vps.annotations import parse_partition, split_summary, validate_splits
records = parse_partition((DATA / 'Eval/list_eval_partition.txt').read_text())
assert split_summary(records) == {
    'train': {'images':25882,'unique_items':3997},
    'query': {'images':14218,'unique_items':3985},
    'gallery': {'images':12612,'unique_items':3985}}
assert validate_splits(records)['query_items_without_gallery_positive'] == 0
print(json.dumps(split_summary(records), indent=2))
'''), md('## Run full experiment\nFull gallery is retained. Bboxes are audited locally and are not used to crop this baseline.\n'), code('''def run(*args):
    subprocess.run([sys.executable, *map(str,args)], check=True)
run('-m','pytest','-q')
run('scripts/pilot.py','--data-root',DATA,'preflight')
run('scripts/pilot.py','--data-root',DATA,'audit','--include-train','--workers','2')
run('scripts/pilot.py','--data-root',DATA,'embed','--batch-size','64','--workers','2','--threads','2')
run('scripts/pilot.py','--data-root',DATA,'evaluate')
run('scripts/pilot.py','--data-root',DATA,'examples','--per-group','10')
run('-m','pip','freeze')
(WORK / 'reports/environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
'''), code('''metrics = json.loads((WORK / 'reports/metrics.json').read_text())
print(json.dumps(metrics, indent=2))
from IPython.display import display, Image
for group in ('success_top1','fail_top1_recovered_top5','fail_top5'):
    for path in sorted((WORK / 'artifacts/examples').glob(group + '_*.png')):
        display(Image(filename=str(path)))
with zipfile.ZipFile(WORK.parent / 'inshop-baseline-results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in ('reports','artifacts','cache/embeddings'):
        for path in sorted((WORK / folder).rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(WORK))
print('Saved results to', WORK.parent / 'inshop-baseline-results.zip')
# Keep the saved cloud output focused on results; inputs remain in the private dataset.
for temporary in (DATA, WORK / '.cache/clip'):
    assert temporary.resolve().is_relative_to(WORK.resolve())
    shutil.rmtree(temporary)
'''), md('''## Interpretation

A hit requires the exact product ID; similar color or clothing category is insufficient.
Balanced example groups are diagnostic and do not represent error prevalence.
Unknown overlap with CLIP's original pretraining data cannot be ruled out.
In-shop studio retrieval does not establish performance on phone photographs.
Do not tune on these test metrics. Choose improvements using train identities.
''')]
    p = ROOT / 'notebooks/deepfashion_clip_baseline.ipynb'
    nb = json.loads(p.read_text(encoding='utf-8'))
    nb['cells'] = cells
    nb['metadata']['kernelspec'] = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
    p.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    meta = {'id':'dangtrung2005/fcvg101-deepfashion-frozen-clip-baseline',
            'title':'FCVG101 DeepFashion frozen CLIP baseline',
            'code_file':'deepfashion_clip_baseline.ipynb', 'language':'python', 'kernel_type':'notebook',
            'is_private':True,'enable_gpu':True,'enable_internet':True,
            'dataset_sources':['dangtrung2005/fcvg101-inshop-private-inputs'],
            'competition_sources':[], 'kernel_sources':[]}
    (p.parent / 'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(p)


if __name__ == '__main__':
    main()
