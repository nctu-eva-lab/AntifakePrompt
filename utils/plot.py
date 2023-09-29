import numpy as np
import matplotlib.pyplot as plt

START_IDX=None
END_IDX=None

Y_LABEL = 'Accuracy (%)'

COLOR = ['#DBCBD8', '#5E4B56', '#FCEC52', '#3B7080', '#E8871E', '#457EAC', '#65743A', '#041F1E']
BAR_WIDTH = 0.17
FIG_WIDTH = 20

BARS = ['SD2', 'SDXL', 'IF', 'DALLE2', 'SGXL', 'ControlNet', 'IP(LaMa)', 'IP(SD2)', 'SR(LTE)', 'SR(SD2)', 'Deeper\nForensics', 'Adverserial\nattack', 'Backdoor\nattack', 'DataPoison\nattack']
LABELS = ['Wang2020', 'DE-FAKE', 'pretrained', 'SD2', 'SD2+IP(balance)', 'SD2+IP(imbalance)', 'SD2+IP+SR', 'SD2+IP+SR+DF']

# 0 for COCO, 1 for Flickr
MODE_IDX = 0
TITLES = ['COCO + X (All)', 'Flickr + X (All)']
FILENAMES = ['COCO_all.png', 'Flickr_all.png']
REAL_ACC = [
    [96.87, 85.97, 98.93, 97.73, 94.53, 90.17, 94.37, 96.17], # COCO
    [96.67, 90.67, 99.63, 95.20, 88.63, 80.70, 92.03, 92.33] # Flickr
]
FAKE_ACC = [
    [00.17, 00.17, 19.17, 03.40, 79.30, 11.87, 07.53, 01.27, 15.27, 01.40, 00.03, 04.93, 15.50, 00.97],
    [97.10, 90.50, 99.20, 68.97, 56.90, 63.97, 15.00, 20.80, 09.33, 32.10, 79.40, 60.40, 22.23, 55.87],
    [39.73, 21.37, 20.13, 40.77, 68.23, 32.80, 10.83, 44.07, 96.87, 67.67, 03.03, 05.20, 03.03, 01.67],
    [98.33, 98.13, 96.57, 88.30, 88.00, 80.50, 06.37, 28.73, 55.00, 60.47, 41.70, 54.10, 44.07, 55.07],
    [97.07, 96.30, 88.83, 98.30, 99.90, 90.40, 40.20, 85.00, 99.97, 99.90, 53.87, 97.53, 94.53, 92.87],
    [98.23, 97.80, 92.77, 98.90, 99.93, 94.33, 49.17, 89.53, 100.0, 99.93, 80.80, 98.40, 96.07, 94.80],
    [95.10, 94.67, 82.70, 99.70, 99.77, 86.17, 36.70, 78.73, 99.93, 99.73, 74.47, 99.27, 94.13, 92.47],
    [95.13, 94.80, 83.77, 99.83, 99.83, 85.90, 33.00, 78.33, 99.93, 99.83, 100.0, 98.10, 91.73, 90.77],
]

# START_IDX=0
# END_IDX=None

def main():
    # Data set
    real_acc = REAL_ACC[MODE_IDX]
    fake_acc = FAKE_ACC
    
    start_idx = START_IDX if START_IDX else 0
    end_idx = END_IDX if END_IDX else len(fake_acc[0])
    bars = BARS[start_idx:end_idx]
    x_base = np.arange(len(bars))*1.6
    
    h = []
    for real_idx, real_i in enumerate(real_acc):
        fake_i = fake_acc[real_idx]
        fake_i = fake_i[start_idx:end_idx]
        mean_acc = []
        for fake_ij in fake_i:
            mean_acc.append((real_i+fake_ij)/2)
        h.append(mean_acc)
    
    plt.figure().set_figwidth(FIG_WIDTH)
    
    x = []
    for i in range(len(h)):
        offset = (-len(h)/2 + i) * BAR_WIDTH
        x_i = [xx+offset for xx in x_base]
        x.append(x_i)
    
    for i in range(len(h)):
        plt.bar(x[i], h[i], color=COLOR[i], width=BAR_WIDTH, align='edge', label=LABELS[i], edgecolor='w')
    
    plt.title(TITLES[MODE_IDX], fontsize=14, fontweight='bold')
    plt.ylabel(Y_LABEL, fontweight='bold')
    plt.xticks(x_base, bars, fontweight='bold')
    plt.ylim([0, 100])
    plt.legend(loc='lower left')
    plt.savefig(FILENAMES[MODE_IDX])

if __name__ == "__main__":
    main()