import os

def main():
    # dirPath = "/eva_data0/denny/textual_inversion/60k_6k_6k/test/0_real/COCO/coco2014_train_224/"
    dirPath = "/eva_data0/denny/textual_inversion/60k_6k_6k/test/0_real/Flickr/flickr30k_224/"
    out_path = "/eva_data0/denny/ControlNet/canny/test/Flickr_path.txt"
    with open(out_path, 'w') as f:
        for _, _, fileNames in os.walk(dirPath):
            for name in sorted(fileNames):
                f.write(f'{os.path.join(dirPath, name)}\n')

if __name__ == '__main__':
    main()