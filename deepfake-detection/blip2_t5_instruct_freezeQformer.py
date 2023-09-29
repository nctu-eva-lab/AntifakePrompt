import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
print(os.environ["CUDA_VISIBLE_DEVICES"])

from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import lavis
from lavis.models import load_model_and_preprocess

import random 
random.seed(43)

class TextInvDataset(Dataset):
    def __init__(self, csv, vis_processors=None, txt_processors=None):
        
        self.path_and_labels = pd.read_csv(csv, index_col="img_path")
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

    def __len__(self):
        
        return len(list(self.path_and_labels.index))

    def __getitem__(self, index):

        image_path = list(self.path_and_labels.index)[index]
        image = Image.open(image_path).convert("RGB")
        if self.vis_processors:
            image = self.vis_processors(image)
        
        label = self.path_and_labels.loc[image_path, "label"]
        
        is_uncommon = "uncommon" in image_path

        return image, label, is_uncommon


# This is for query lost of images

from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import time

class InstructBLIP():
    def __init__(self, name="blip2_vicuna_instruct_textinv_freezeQformer", model_type="vicuna7b", is_eval=True, device="cpu") -> None:
        print(f'Loading model...')
        #self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name, model_type, is_eval, device)
        self.imgs = []
        self.labels = []
        
        # QA
        self.question = ""
        
        # results
        self.acc = None
        self.confusion_mat = None
        
        self.acc_3class = None
        self.confusion_mat_3class = None
        
        self.com_acc = None
        self.com_confusion_mat = None
        self.uncom_acc = None
        self.uncom_confusion_mat = None

    def LoadModels(self, model, vis_processors, txt_processors, device):
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.device = device
    
    def LoadImages(self, dir, num):
        onlyfiles = []
        
        for f in sorted(listdir(dir)):
            if isfile(join(dir, f)):
                onlyfiles.append(join(dir, f))
        
        onlyfiles = random.sample(onlyfiles, num)
        
        raw_img_list = []
        with tqdm(total=len(onlyfiles), desc=f'Loading imgs from {dir}') as pbar:
            for f in onlyfiles:
                raw_img = Image.open(f).convert("RGB")
                raw_img_list.append(raw_img)
                pbar.update(1)
        
        return raw_img_list

    def LoadData(self, real_dir, fake_dir, num=1000):
        #real_imgs = LoadImages(join(root_dir, "0_real"))
        #fake_imgs = LoadImages(join(root_dir, "1_fake"))
        real_imgs = self.LoadImages(real_dir, num)
        fake_imgs = self.LoadImages(fake_dir, num)
        
        self.imgs = real_imgs + fake_imgs
        self.labels = [0]*len(real_imgs) + [1]*len(fake_imgs)
        #return self.imgs, self.labels
      
    def LoadData_batch(self, csv_path):
        self.csv = csv_path
        self.dataset = TextInvDataset(csv=csv_path, vis_processors=self.vis_processors["eval"])
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=8, shuffle=False, num_workers=8)    
        
    def LoadData3Class(self, real_dir, fake_common_dir, fake_uncommon_dir, num=[1000, 500, 500]):
        #real_imgs = LoadImages(join(root_dir, "0_real"))
        #fake_imgs = LoadImages(join(root_dir, "1_fake"))
        self.num = num
        real_imgs = self.LoadImages(real_dir, num[0])
        fake_common_imgs = self.LoadImages(fake_common_dir, num[1])
        fake_uncommon_imgs = self.LoadImages(fake_uncommon_dir, num[2])
        
        self.imgs = real_imgs + fake_common_imgs + fake_uncommon_imgs
        self.labels = [0]*len(real_imgs) + [1]*(len(fake_common_imgs)+len(fake_uncommon_imgs))
        self.label_3class = [0]*len(real_imgs) + [1]*len(fake_common_imgs) + [2]*len(fake_uncommon_imgs)
        #return self.imgs, self.labels, self.label_3class

    def QueryImgs(self, question, true_string="yes"):
        self.ans_list = []
        self.question = question
        
        with tqdm(total=len(self.imgs), desc=f'Answering') as pbar:
            for idx, img in enumerate(self.imgs):
                image = self.vis_processors["eval"](img).unsqueeze(0).to(self.device)

                samples = {"image": image, "text_input": question}
                
                ans = self.model.predict_answers(samples=samples, inference_method="generate")[0]
                self.ans_list.append(0 if ans == true_string else 1)
                
                pbar.update(1)
        
        self.acc = accuracy_score(self.labels, self.ans_list)
        self.confusion_mat = confusion_matrix(self.labels, self.ans_list)
        
        self.PrintResult()
        
        return self.acc, self.confusion_mat, self.ans_list
    
    def QueryImgs_batch(self, question, true_string="yes", logPath='log.txt', question_Qformer="Is this photo real?"):
        self.labels = []
        self.label_3class = []
        self.ans_list = []
        self.question = question
        self.question_Qformer = question_Qformer
        
        for image, label, is_uncommon in tqdm(self.dataloader):
            
            image = image.to(self.device)
            
            questions = [self.question] * image.shape[0]
            question_Qformer = [self.question_Qformer] * image.shape[0]
            samples = {"image": image, "text_input": questions, "text_input_Qformer": question_Qformer}
            
            ans = self.model.predict_answers(samples=samples, inference_method="generate", answer_list=["yes", "no"])
            pred_label = [0 if a == true_string else 1 for a in ans]
            self.ans_list += pred_label
            
            label = [0 if l == true_string else 1 for l in label]
            self.labels += label
            
            label_3class = label.copy()
            label_3class = [2 if is_uncommon[idx] else l for idx, l in enumerate(label)]
            
            self.label_3class += label_3class
        
        self.acc = accuracy_score(self.labels, self.ans_list)
        self.confusion_mat = confusion_matrix(self.labels, self.ans_list, labels=[0,1])
        
        self.ans_list = np.array(self.ans_list)
        self.labels = np.array(self.labels)
        self.label_3class = np.array(self.label_3class)
        
        self.PrintResult(three_class=True, logPath=logPath)
        
        return self.acc, self.confusion_mat, self.ans_list, self.labels, self.label_3class
    
    def Query(self, image, question):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        
        samples = {"image": image, "text_input": question}
        ans = self.model.predict_answers(samples=samples, inference_method="generate")[0]
        return ans

    def PrintResult(self, three_class=False, acc=None, confusion_mat=None, ans_list=None, labels=None, label_3class=None, logPath=None):
        
        if acc:
            self.acc = acc
        if confusion_mat:
            self.confusion_mat = confusion_mat
        if ans_list:
            self.ans_list = ans_list
        if labels:
            self.labels = labels
        if label_3class:
            self.label_3class = label_3class
        
        if logPath:
            logfile = open(logPath, 'a')
        
        if three_class:
            #assert type(self.num) == list, "Type of num should be list."
            
            print(f'[TIME]      : {time.ctime()}', file=logfile)
            print(f'[Finetuned] : {self.model.finetuned}', file=logfile)
            print(f'[Data csv]  : {self.csv}', file=logfile)
            print(f'[Question]  : {self.question}\n', file=logfile)
            
            print(f'=== Overall ===', file=logfile)
            print(f'Acc: {self.acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.confusion_mat, logfile=logfile)
            print('\n', file=logfile)
            
            real_ans_list = self.ans_list[self.label_3class==0]
            real_label = [0] * len(real_ans_list)
            self.real_acc = accuracy_score(real_label, real_ans_list)
            self.real_confusion_mat = confusion_matrix(real_label, real_ans_list, labels=[0,1])
            print(f'=== Real images ===', file=logfile)
            print(f'Acc: {self.real_acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.real_confusion_mat, logfile=logfile)
            print('\n', file=logfile)
            
            com_ans_list = self.ans_list[self.label_3class==1]
            com_label = [1] * len(com_ans_list)
            self.com_acc = accuracy_score(com_label, com_ans_list)
            self.com_confusion_mat = confusion_matrix(com_label, com_ans_list, labels=[0,1])
            print(f'=== Common fake images ===', file=logfile)
            print(f'Acc: {self.com_acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.com_confusion_mat, logfile=logfile)
            print('\n', file=logfile)
            
            uncom_ans_list = self.ans_list[self.label_3class==2]
            uncom_label = [1] * len(uncom_ans_list)
            self.uncom_acc = accuracy_score(uncom_label, uncom_ans_list)
            self.uncom_confusion_mat = confusion_matrix(uncom_label, uncom_ans_list, labels=[0,1])
            print(f'=== Uncommon fake images ===', file=logfile)
            print(f'Acc: {self.uncom_acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.uncom_confusion_mat, logfile=logfile)
            print('\n', file=logfile)
        else:
            print(f'Question: {self.question}\n', file=logfile)
            print(f'Acc: {self.acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.confusion_mat, logfile=logfile)
            print('\n', file=logfile)
        
        logfile.close()
    
    def PrintConfusion(self, mat, logfile):
        padding = ' '
        print(f'        | Pred real | Pred fake |', file=logfile)
        print(f'GT real | {mat[0, 0]:{padding}<{10}}| {mat[0, 1]:{padding}<{11}}|', file=logfile)
        print(f'GT fake | {mat[1, 0]:{padding}<{10}}| {mat[1, 1]:{padding}<{11}}|', file=logfile)
        
    def MultipleAns(self, ans1, ans2):
    
        # Q1: Is this photo common in real world?
        # Q2: Is this photo generated by a model?
        
        final_ans = []
        for ans in zip(ans1, ans2):
            if ans[0] == 0 and ans[1] == 0:
                final_ans.append(0)
            else:
                final_ans.append(1)
        
        acc = accuracy_score(self.labels, final_ans)
        confusion_mat = confusion_matrix(self.labels, final_ans)
        print(f'Accuracy: {acc*100:.2f}%')
        self.PrintConfusion(confusion_mat)
        
        self.ans_list = final_ans
        self.acc = acc
        self.confusion_mat = confusion_mat
        
        return acc, confusion_mat, final_ans
    
def print_combine_result(pretrained_ans, finetuned_ans, label, logPath):
    
    logfile = open(logPath, 'a')
    
    def _print_confusion(mat, logfile):
        padding = ' '
        print(f'        | Pred real | Pred fake |', file=logfile)
        print(f'GT real | {mat[0, 0]:{padding}<{10}}| {mat[0, 1]:{padding}<{11}}|', file=logfile)
        print(f'GT fake | {mat[1, 0]:{padding}<{10}}| {mat[1, 1]:{padding}<{11}}|', file=logfile)
    
    comb_ans = np.ceil((pretrained_ans + finetuned_ans)/2).astype(np.int64)
    
    comb_acc = accuracy_score(label, comb_ans)
    comb_confusion_mat = confusion_matrix(label, comb_ans, labels=[0,1])
    
    print(f'=== Overall (Comb) ===', file=logfile)
    print(f'Acc: {comb_acc*100:.2f}%', file=logfile)
    _print_confusion(comb_confusion_mat, logfile=logfile)
    print('\n', file=logfile)
    
    real_ans_list = comb_ans[label==0]
    real_label = [0] * len(real_ans_list)
    real_acc = accuracy_score(real_label, real_ans_list)
    real_confusion_mat = confusion_matrix(real_label, real_ans_list, labels=[0,1])
    print(f'=== Real images (Comb) ===', file=logfile)
    print(f'Acc: {real_acc*100:.2f}%', file=logfile)
    _print_confusion(real_confusion_mat, logfile=logfile)
    print('\n', file=logfile)
    
    
    com_ans_list = comb_ans[label==1]
    com_label = [1] * len(com_ans_list)
    com_acc = accuracy_score(com_label, com_ans_list)
    com_confusion_mat = confusion_matrix(com_label, com_ans_list, labels=[0,1])
    print(f'=== Common fake images (Comb) ===', file=logfile)
    print(f'Acc: {com_acc*100:.2f}%', file=logfile)
    _print_confusion(com_confusion_mat, logfile=logfile)
    print('\n', file=logfile)
    
    return comb_acc, comb_confusion_mat, comb_ans

def main():
    
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/log.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SDXL_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/IF_postfix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_imbalance_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_balance_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_balance_prefix_onlyCommon.txt'
    # logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_balance_replace_onlyCommon.txt'
    
    logPath = '/home/denny/LAVIS/deepfake-detection/log/SD2_SD2IP_90k_postfix_freezeQformer_onlyCommon.txt'
    
    
    q1 = "Is this photo real?"
    q2 = "Is this photo real [*]?"
    # q2 = "[*] Is this photo real?"
    # q2 = "Is this photo [*]?"
    
    q_Qformer = "Is this photo real?"
    
    file = open(logPath, 'a')
    file.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct_textinv_freezeQformer", model_type="vicuna7b", is_eval=True, device=device)
    
    print(f'Load model OK!')
    
    instruct = InstructBLIP()
    instruct.LoadModels(model, vis_processors, txt_processors, device)
    
    print(f'Log path: {logPath}')
    print(f'Q1: {q1}')
    print(f'Q2: {q2}')
    
    csvfiles = [
        # "/eva_data0/denny/textual_inversion/debug_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_COCO_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_Flickr_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_SD2_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_SDXL_label.csv", 
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_IF_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_DALLE_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_SGXL_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_Control_COCO_label.csv",
        # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_lama_label.csv",
        # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_SD2IP_label.csv",
        # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_lte_label.csv",
        # "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_SD2SR_label.csv",
        "/eva_data0/iammingggg/textual_inversion/60k_6k_6k/test_deeperforensics_faceOnly_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_AdvAtk_Imagenet_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_Backdoor_Imagenet_label.csv",
        # "/eva_data0/denny/textual_inversion/60k_6k_6k/test_DataPoison_Imagenet_label.csv",
        ]
    
    for csv_path in csvfiles:
        instruct.LoadData_batch(csv_path=csv_path)
        print(f'Load data from {csv_path}')
        
        question = q1
        acc, confusion_mat, pretrained_ans_list, labels, label_3class = instruct.QueryImgs_batch(question=question, question_Qformer=q_Qformer, true_string="yes", logPath=logPath)
        print(f'Question: {question}')
        print(f'Acc: {acc*100:.2f}%')

        question = q2
        acc, confusion_mat, finetuned_ans_list, labels, label_3class = instruct.QueryImgs_batch(question=question, question_Qformer=q_Qformer, true_string="yes", logPath=logPath)
        print(f'Question: {question}')
        print(f'Acc: {acc*100:.2f}%')
        
        comb_acc, comb_confusion_mat, comb_ans = print_combine_result(pretrained_ans_list, finetuned_ans_list, labels, logPath=logPath)
        print(f'[Combination]')
        print(f'Acc: {comb_acc*100:.2f}%')
        

if __name__ == '__main__':
    main()