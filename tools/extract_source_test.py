import os, random, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='data/combined_train_pcap')
parser.add_argument('--dst', default='data/test_ustc')
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--prefix', type=str, default='ustc_')
args = parser.parse_args()

random.seed(42)
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_root = os.path.join(project_root, args.src)
dst_root = os.path.join(project_root, args.dst)

moved = { 'Benign': [], 'Malware': [] }
summary = {}
for cls in ['Benign', 'Malware']:
    cls_dir = os.path.join(src_root, cls)
    if not os.path.isdir(cls_dir):
        summary[cls] = {'total':0,'pref_total':0,'moved':0}
        continue
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.pcap')]
    pref_files = [f for f in files if f.lower().startswith(args.prefix.lower())]
    summary[cls] = {'total': len(files), 'pref_total': len(pref_files)}
    k = max(1, int(len(pref_files) * args.ratio)) if len(pref_files)>0 else 0
    to_move = random.sample(pref_files, k) if k>0 else []
    target_cls_dir = os.path.join(dst_root, cls)
    os.makedirs(target_cls_dir, exist_ok=True)
    for fn in to_move:
        srcp = os.path.join(cls_dir, fn)
        dstp = os.path.join(target_cls_dir, fn)
        shutil.move(srcp, dstp)
        moved[cls].append(fn)
    summary[cls]['moved'] = len(to_move)

print('Summary:')
for cls, v in summary.items():
    print(f"  {cls}: total_files={v['total']}, pref_total={v['pref_total']}, moved={v['moved']}")

print('\nMoved files list:')
for cls in ['Benign','Malware']:
    print(f'-- {cls} ({len(moved[cls])}) --')
    for fn in moved[cls]:
        print(fn)
