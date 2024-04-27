import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score, precision_recall_curve
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def find_best_threshold(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    f1_scores = [f1_score(y_true, y_pred > threshold) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"roc curve threshold: {best_threshold}")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = [f1_score(y_true, y_pred >= threshold) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]

    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, marker='.', label='Precision-Recall Curve')
    plt.scatter(recalls[np.argmax(f1_score)], precisions[np.argmax(f1_score)], s=100, c='red', label='Best threshold (F1 Score)')

    target_precisions = [0.75, 0.8]
    colors = ['green', 'blue']
    for target_precision, color in zip(target_precisions, colors):
        # Find the index closest to the target precision
        closest_precision_index = np.argmin(np.abs(precisions - target_precision))
        closest_threshold = thresholds[closest_precision_index]

        # Plot the corresponding recall and precision as a large dot
        plt.scatter(recalls[closest_precision_index], precisions[closest_precision_index], s=100, c=color, label=f'Precision {target_precision}')

        # Annotate the threshold and precision point
        plt.annotate(f'Threshold: {closest_threshold:.2f}\nPrecision: {precisions[closest_precision_index]:.2f}',
                    (recalls[closest_precision_index], precisions[closest_precision_index]),
                    textcoords="offset points", xytext=(10,10), ha='center')
        

        # Find the index for the threshold closest to 0.3
        specific_threshold = 0.35
        closest_index = np.argmin(np.abs(thresholds - specific_threshold))
        specific_precision = precisions[closest_index]
        specific_recall = recalls[closest_index]

        # Plot the corresponding recall and precision as a large dot
        plt.scatter(specific_recall, specific_precision, s=100, c='orange', label=f'Threshold 0.3')

        # Annotate the threshold and precision point
        plt.annotate(f'Threshold: {thresholds[closest_index]:.2f}\nPrecision: {specific_precision:.2f}',
                    (specific_recall, specific_precision),
                    textcoords="offset points", xytext=(10,10), ha='center')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('UniversalFakeDetect PR Curve: CelebA-HQ')
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.savefig("/home/ubuntu/hannah/UniversalFakeDetect/clip_vitl14/pr-curve.png")
    
    
    
    
    return best_threshold
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def calculate_acc(y_true, y_pred, thres):
    results_dict = {}
    results_dict['r_acc'] = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    results_dict['f_acc'] = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    results_dict['acc'] = accuracy_score(y_true, y_pred > thres)

    binary_preds = [int(i > thres) for i in y_pred]
    results_dict['tn'], results_dict['fp'], results_dict['fn'], results_dict['tp'] = confusion_matrix(y_true, binary_preds).ravel()

    results_dict['prec'] = precision_score(y_true, binary_preds)
    results_dict['recall'] = recall_score(y_true, binary_preds)
    results_dict['f1'] = f1_score(y_true, binary_preds)

    return results_dict

def validate(model, loader, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader)))
        for img, label in loader:
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(f"total number of samples in dataset: {len(y_pred)}")

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    results_dict_default = calculate_acc(y_true, y_pred, 0.5)
    results_dict_default['ap'] = ap
    if not find_thres:
        return results_dict_default

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    results_dict_best_thres = calculate_acc(y_true, y_pred, best_thres)
    results_dict_best_thres['best_thres'] = best_thres

    return results_dict_default, results_dict_best_thres
    
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[-1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)

    return image_list


class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None):

        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])


    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        print(f"real samples: {len(real_list)}")
        print(f"fake samples: {len(fake_list)}")
        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

    opt = parser.parse_args()

    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    # need following lines for trained model weight architecture, if it's saved the whole model
    # state_dict = torch.load(opt.ckpt, map_location='cpu')['model']
    # state_dict = {"weight": state_dict["fc.weight"], "bias": state_dict["fc.bias"]}
    
    # save state dict for the last layer
    # torch.save(state_dict, opt.ckpt)

    model.fc.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda()

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode) ]



    for dataset_path in (dataset_paths):
        set_seed()

        dataset = RealFakeDataset(  dataset_path['real_path'], 
                                    dataset_path['fake_path'], 
                                    dataset_path['data_mode'], 
                                    opt.max_sample, 
                                    opt.arch,
                                    jpeg_quality=opt.jpeg_quality, 
                                    gaussian_sigma=opt.gaussian_sigma,
                                    )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        results_dict_default, results_dict_best_thres = validate(model, loader, find_thres=True)

        with open( os.path.join(opt.result_folder,'stats.txt'), 'a') as f:
            f.write('average precision: ' + str(round(results_dict_default['ap']*100, 2))+'\n' )
            f.write('accuracy (0.5 threshold): ' + str(round(results_dict_default['r_acc']*100, 2))+'  '+str(round(results_dict_default['f_acc']*100, 2))+'  '+str(round(results_dict_default['acc']*100, 2))+'\n' )
            f.write(f"stats (0.5 threshold): TN: {results_dict_default['tn']}, FP: {results_dict_default['fp']}, FN: {results_dict_default['fn']}, TP {results_dict_default['tp']}\n")
            f.write(f"f1: {results_dict_default['f1']}, prec: {results_dict_default['prec']}, recall: {results_dict_default['recall']}\n")
            f.write(f"accuracy (best threshold {results_dict_best_thres['best_thres']}): " + str(round(results_dict_best_thres['r_acc']*100, 2))+'  '+str(round(results_dict_best_thres['f_acc']*100, 2))+'  '+str(round(results_dict_best_thres['acc']*100, 2))+'\n' )
            f.write(f"stats (best threshold {results_dict_best_thres['best_thres']}): TN: {results_dict_best_thres['tn']}, FP: {results_dict_best_thres['fp']}, FN: {results_dict_best_thres['fn']}, TP {results_dict_best_thres['tp']}\n")
            f.write(f"f1: {results_dict_best_thres['f1']}, prec: {results_dict_best_thres['prec']}, recall: {results_dict_best_thres['recall']}\n")


