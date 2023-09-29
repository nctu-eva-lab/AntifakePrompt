import pandas as pd
import argparse
import os

EXTENSIONS = [".png", ".jpg", ".jpeg"]

def valid_path(args, img_path):
    if os.path.splitext(img_path)[-1].lower() not in EXTENSIONS:
        return False

    if args.real_dir and args.real_dir in img_path:
        return True
    elif args.fake_dir and args.fake_dir in img_path:
        return True
    
    return False
        

def arg_parser():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--img_root", type=str, default=None)
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--fake_dir", type=str, default=None)
    parser.add_argument("--real_label", type=str, default="yes")
    parser.add_argument("--fake_label", type=str, default="no")
    # parser.add_argument("--label", type=str)
    parser.add_argument("--out", type=str, default="label.csv")
    
    return parser.parse_args()

def main():
    args = arg_parser()
    assert args.out.endswith(".csv"), "Output path should be a path to a csv file."
    
    if args.real_dir:
        # real_path = os.path.join(args.img_root, args.real_dir)
        real_path = args.real_dir
        assert os.path.exists(real_path), f'Real path {real_path} does not exist.'
    
    if args.fake_dir:
        # fake_path = os.path.join(args.img_root, args.fake_dir)
        fake_path = args.fake_dir
        assert os.path.exists(fake_path), f'Fake path {fake_path} does not exist.'
    
    img_paths = []
    labels = []
    n_real_imgs = 0
    n_fake_imgs = 0
    
    if args.real_dir:
        for root, dirs, files in os.walk(real_path):
            for f in files:
                img_path = os.path.join(root, f)
                
                if not valid_path(args, img_path):
                    continue
                
                img_paths.append(img_path)
                labels.append(args.real_label)
                
                n_real_imgs += 1
        
        print(f'Found {n_real_imgs} real images in [{real_path}]')

    if args.fake_dir:
        for root, dirs, files in os.walk(fake_path):
            for f in files:
                img_path = os.path.join(root, f)
                
                if not valid_path(args, img_path):
                    continue
                
                img_paths.append(img_path)
                labels.append(args.fake_label)
                
                n_fake_imgs += 1
        print(f'Found {n_fake_imgs} fake images in [{fake_path}]')
    
    assert len(img_paths) == len(labels), "Length of img path should be equal to labels."
    
    paths_and_labels = pd.DataFrame({
        "img_path": img_paths,
        "label": labels
    })
    
    paths_and_labels.to_csv(args.out)
    print(f'Write to [{args.out}]')
    
if __name__ == "__main__":
    main()
    