import pandas as pd
import numpy as np
import os
import time
import random 
import argparse
from PIL import Image
from os import makedirs
from os.path import dirname
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import lavis
from lavis.models import load_model_and_preprocess

random.seed(43)

EXT = ['.jpg', '.jpeg', '.png']

class TextInvDataset(Dataset):
    def __init__(self, roots, labels, vis_processors=None, txt_processors=None):
        
        self.path_and_labels = {
            'img_path': [],
            'label': []
        }
        
        assert len(roots) == len(labels), "Please assign a label for each image root."
        
        for root, label in zip(roots, labels):
            n_sample = 0
            for r, dirs, files in os.walk(root):
                for file in files:
                    if os.path.splitext(file)[-1] in EXT:
                        self.path_and_labels["img_path"].append(os.path.join(r, file))
                        self.path_and_labels["label"].append(label)
                        n_sample += 1
            print(f'Found {n_sample} images with label "{label}".')
        
        self.path_and_labels = pd.DataFrame.from_dict(self.path_and_labels)
        self.path_and_labels.set_index("img_path", inplace=True)
        
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

        return image, label

class InstructBLIP():
    def __init__(self, name="blip2_vicuna_instruct_textinv", model_type="vicuna7b", is_eval=True, device="cpu") -> None:
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
        
    def LoadData(self, roots, labels):
        self.roots = [roots] if isinstance(roots, str) else roots
        self.text_labels = [labels] if isinstance(labels, str) else labels
        self.dataset = TextInvDataset(self.roots, self.text_labels, vis_processors=self.vis_processors["eval"])
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=8, shuffle=False, num_workers=8)    
    
    def QueryImgs_batch(self, question, true_string="yes", logPath='log.txt'):
        self.labels = []
        self.label_3class = []
        self.ans_list = []
        self.question = question
        
        for image, label in tqdm(self.dataloader):
            
            image = image.to(self.device)
            
            questions = [self.question] * image.shape[0]
            
            # samples = {"image": image, "text_input": questions}
            # ans = self.model.predict_answers(samples=samples, inference_method="generate", answer_list=["yes", "no"])
            # pred_label = [0 if a == true_string else 1 for a in ans]
            
            samples = {"image": image, "prompt": questions}
            candidates = ["yes", "no"]
            ans = self.model.predict_class(samples=samples, candidates=candidates)
            pred_label = [0 if candidates[list(a).index(0)]==true_string else 1 for a in ans]
            self.ans_list += pred_label
            
            label = [0 if l == true_string else 1 for l in label]
            self.labels += label
        
        self.acc = accuracy_score(self.labels, self.ans_list)
        self.confusion_mat = confusion_matrix(self.labels, self.ans_list, labels=[0,1])
        
        self.ans_list = np.array(self.ans_list)
        self.labels = np.array(self.labels)
        self.label_3class = np.array(self.label_3class)
        
        self.PrintResult(detailed=True, logPath=logPath)
        
        return self.acc, self.confusion_mat, self.ans_list, self.labels, self.label_3class
    
    def Query(self, image, question):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        
        samples = {"image": image, "prompt": question}
        candidates = ["yes", "no"]
        ans = self.model.predict_class(samples=samples, candidates=candidates)
        pred_label = ["Real" if candidates[list(a).index(0)]=="yes" else "Fake" for a in ans]
        return pred_label

    def PrintResult(self, detailed=False, acc=None, confusion_mat=None, ans_list=None, labels=None, logPath=None):
        
        if acc:
            self.acc = acc
        if confusion_mat:
            self.confusion_mat = confusion_mat
        if ans_list:
            self.ans_list = ans_list
        if labels:
            self.labels = labels
        
        if logPath:
            logfile = open(logPath, 'a')
        
        if detailed:
            
            print(f'[TIME]      : {time.ctime()}', file=logfile)
            print(f'[Finetuned] : {self.model.finetuned}', file=logfile)
            print(f'[Img roots] : {self.roots}', file=logfile)
            print(f'[Labels]    : {self.text_labels}', file=logfile)
            print(f'[Question]  : {self.question}\n', file=logfile)
            
            print(f'=== Overall ===', file=logfile)
            print(f'Acc: {self.acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.confusion_mat, logfile=logfile)
            print('\n', file=logfile)
            
            if 0 in self.labels:
                real_ans_list = self.ans_list[self.labels==0]
                real_label = [0] * len(real_ans_list)
                self.real_acc = accuracy_score(real_label, real_ans_list)
                self.real_confusion_mat = confusion_matrix(real_label, real_ans_list, labels=[0,1])
                print(f'=== Real images ===', file=logfile)
                print(f'Acc: {self.real_acc*100:.2f}%', file=logfile)
                self.PrintConfusion(self.real_confusion_mat, logfile=logfile)
                print('\n', file=logfile)
            else:
                print(f'=== No real images ===\n', file=logfile)
            
            
            if 1 in self.labels:
                fake_ans_list = self.ans_list[self.labels==1]
                fake_label = [1] * len(fake_ans_list)
                self.com_acc = accuracy_score(fake_label, fake_ans_list)
                self.com_confusion_mat = confusion_matrix(fake_label, fake_ans_list, labels=[0,1])
                print(f'=== Fake images ===', file=logfile)
                print(f'Acc: {self.com_acc*100:.2f}%', file=logfile)
                self.PrintConfusion(self.com_confusion_mat, logfile=logfile)
                print('\n', file=logfile)
            else:
                print(f'=== No fake images ===\n', file=logfile)
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
    
def arg_parser():
    parser = argparse.ArgumentParser()

    #parser.add_argument('--question', type=str, nargs='+', help='The prompt question.')
    parser.add_argument('--img_path', type=str, default=None, help='The path to the target image.')
    parser.add_argument('--real_dir', type=str, default=None, help='The path to the real directory.')
    parser.add_argument('--fake_dir', type=str, default=None, help='The path to the fake directory.')
    parser.add_argument('--real_label', type=str, default="yes", help='The label for real images.')
    parser.add_argument('--fake_label', type=str, default="no", help='The label for fake images.')
    parser.add_argument('--log', type=str, default="log/log.txt", help='Path to the log file.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='blip2_vicuna_instruct_textinv')
    parser.add_argument('--model_type', type=str, default='vicuna7b')

    return parser.parse_args()

def main():
    
    args = arg_parser()
    
    #question = ' '.join(args.question)
    question = "Is this photo real [*]?"
    logPath = args.log
    device = args.device
    model_name = args.model_name
    model_type=args.model_type
    
    if dirname(logPath):
        makedirs(dirname(logPath), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and device=="cuda" else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
    
    print(f'Load model OK!')
    
    instruct = InstructBLIP()
    instruct.LoadModels(model, vis_processors, txt_processors, device)
    
    print(f'Log path: {logPath}')
    print(f'Question: {question}')
    
    if args.img_path:
        # Query only one image
        image = Image.open(args.img_path)
        ans = instruct.Query(image, question)
        print(ans[0])
    
    elif args.real_dir and args.fake_dir:
        # Query batch of images
        roots = [args.real_dir, args.fake_dir]
        labels = [args.real_label, args.fake_label]
        instruct.LoadData(roots, labels)
        
        acc, confusion_mat, pretrained_ans_list, labels, label_3class = instruct.QueryImgs_batch(question=question, true_string="yes", logPath=logPath)
        print(f'Question: {question}')
        print(f'Acc: {acc*100:.2f}%')
    else:
        # Only real or fake directory
        roots = [args.real_dir if args.real_dir else args.fake_dir]
        labels = [args.real_label if args.real_label else args.fake_label]
        instruct.LoadData(roots, labels)
        
        acc, confusion_mat, pretrained_ans_list, labels, label_3class = instruct.QueryImgs_batch(question=question, true_string="yes", logPath=logPath)
        print(f'Question: {question}')
        print(f'Acc: {acc*100:.2f}%')
        
        

if __name__ == '__main__':
    main()