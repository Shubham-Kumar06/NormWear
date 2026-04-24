import pickle, torch, numpy as np, os, sys
from multiprocessing import Pool
from tqdm import tqdm

sys.path.insert(0, ".")
from pretrain_pipeline.dataset import cwt_wrap

input_dir  = "data/pretrain/wearable_pretrain"
output_dir = "data/pretrain/wearable_pretrain_cwt"

def process_file(fn):
    out_path = os.path.join(output_dir, fn)
    if os.path.exists(out_path):
        return
        
    try:
        with open(os.path.join(input_dir, fn), "rb") as f:
            d = pickle.load(f)

        tss = d['tss']
        if isinstance(tss, torch.Tensor):
            tss = tss.float()
        else:
            tss = torch.from_numpy(tss).float()

        cwt = cwt_wrap(tss)  # [nvar, 3, L, 65]

        out = {
            'tss': d['tss'] if isinstance(d['tss'], np.ndarray) else d['tss'].numpy(),
            'cwt': cwt.permute(0, 2, 3, 1).numpy()  # [nvar, L, 65, 3]
        }

        with open(out_path, "wb") as f:
            pickle.dump(out, f)
    except Exception as e:
        print(f"Error processing {fn}: {e}")

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".pkl")]
    print(f"Found {len(files)} files to process...")

    with Pool() as p:
        list(tqdm(p.imap_unordered(process_file, files), total=len(files)))
    
    print("Preprocessing complete!")