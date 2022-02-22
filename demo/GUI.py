

from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import math as m
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk

import os
import sys

root_path = os.path.abspath(os.path.join('..'))
sys.path.append(root_path)
import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
import cv2
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.utils.config import voc320, MEANS
from libs.data_layers.transform import base_transform
from matplotlib import pyplot as plt

import pdb

is_gpu = False
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    is_gpu = True

# for VOC
class_names = ['__background__',  # always index 0
                         'buffalo', 'elephant', 'gazellegrants',
                         'gazellethomsons','impala',
                         'giraffe', 'koribustard', 'lionmale',
                         'lionfemale', 'wildebeest', 'zebra']
num_classes = len(class_names)

win = tk.Tk()
win.geometry("700x900+400+100")
win.resizable(width=True, height=True)
win.title("A Wildlife Recognition System")

# frm = tk.Frame(win)
# frm.pack()
frm_top = tk.Frame(win)
frm_top_left = tk.Frame(win)
frm_top_right = tk.Frame(win)
frm_bottom_left = tk.Frame(win)
frm_bottom_right = tk.Frame(win)
# frm_right = tk.Frame(frm)
# frm_bottom = tk.Frame(frm)

frm_top.grid(row = 0, column = 0, padx = 5, pady = 5)
frm_top_left.grid(row = 1, column = 0, padx = 5, pady = 5)
frm_top_right.grid(row = 1, column = 1, padx = 5, pady = 5)
frm_bottom_left.grid(row = 2, column = 0, padx = 5, pady = 5)
frm_bottom_right.grid(row = 2, column = 1, padx = 5, pady = 5)

# frm_right.pack(side = "right")
# frm_bottom.pack(side = "bottom")

panel_input = tk.Label(frm_top_left, image = None)
panel_predict = tk.Label(frm_top_right, image = None)
#panel_concat = tk.Label(frm_bottom_left, image = None)

#panel_concat.pack(side="bottom")

panel_input.grid(row = 0, column = 0, padx = 5, pady = 5)
panel_predict.grid(row = 0, column = 1, padx = 5, pady = 5)
#panel_concat.grid(row = 0, column = 0, padx = 5, pady = 5)

cfg = voc320
base_network = 'vgg16'
model_path = '../output/vgg16_refinedet320_voc_40000.pth'
print('Construct {}_refinedet network.'.format(base_network))
refinedet = VGGRefineDet(cfg['num_classes'], cfg)
refinedet.create_architecture()
# for CPU
net = refinedet
# for GPU
if is_gpu:
    net = refinedet.cuda()
    cudnn.benchmark = True
# load weights
net.load_weights(model_path)
net.eval()

check = 0

def resize(img):
    
    img_size = img.size
    ratio = img_size[1]/img_size[0]
    size = 900
    if img_size[1] > img_size[0]:
        img = img.resize((round(size*(1/ratio)), size), Image.ANTIALIAS)
    else:
        img = img.resize((size, round(size*ratio)), Image.ANTIALIAS)
    
    return img


def openfile():    
    global file_path    
    
    # destroy_frm_left()
    # destroy_frm_right()
    
    file_path = filedialog.askopenfilename()
    destroy_frm_bottom_right()
    img = Image.open(file_path)
    
    image = resize(img)
    
    img_input = ImageTk.PhotoImage(image)
    
    panel_input.config(image = '')
    panel_predict.config(image = '')
    #panel_concat.config(image = '')
    
    panel_input.image = img_input
    panel_input.config(image = img_input)
    
    #how to put your result in panel
    
    
    
    #panel_concat.image = img_input
    #panel_concat.config(image = img_input)
    #end
    
    # panel_input.pack()
    # panel_predict.pack()
    # panel_concat.pack()

def destroy_frm_left(): 
    global frm_left
    global panel_input
    
    frm_left.destroy()
    frm_left = tk.Frame(frm)
    frm_left.pack(side = "left")
    panel_input = tk.Label(frm_left, image = None)


def destroy_frm_bottom_right(): 
    global frm_bottom_right
    
    frm_bottom_right.destroy()
    frm_bottom_right = tk.Frame(win)
    frm_bottom_right.grid(row = 2, column = 1, padx = 5, pady = 5)
    #panel_predict = tk.Label(frm_right, image = None)
    
def img_bottom(img):
    #img_input = Image.fromarray(img)
    
    img_concat = resize(img)
    img_concat = ImageTk.PhotoImage(img_concat)
    panel_predict.image = img_concat
    panel_predict.config(image = img_concat)
    
def img_right(img):
    #img_predict = Image.fromarray(img)
    
    img_predict = resize(img)
    img_predict = ImageTk.PhotoImage(img_predict)
    
    panel_predict.config(image = '')
    panel_predict.config(image = img_predict)
    panel_predict.image = img_predict
    panel_predict.pack()
    
def _from_rgb(rgb):

    return "#%02x%02x%02x" % rgb   

def detection():
    global save_path
    global label_img
    global check
    '''
    check = 0    
    num=1
    testGene = testGenerator(file_path, num)
    results = model.predict_generator(testGene,num,verbose=1)
    save_path = "./output/"+file_path.rsplit("/", 1)[1]
    label = saveResult_1(save_path,results)
    img = Image.open(save_path)
    img_right(img)
    
    img = cv2.imread(file_path)
    label = np.uint8(label)
    label_img = np.uint8(label)
    
    label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros((label.shape[0], label.shape[1]), dtype=int)
    mask = mask+255
    
    mask[label_gray==255]=0
    print(mask.shape)
    for i in range(0,3):
        tmp = img[:,:,i] - mask
        tmp[tmp<0] = 0
        img[:,:,i] = tmp
    mask = 255 - mask
    for i in range(0,3):
        tmp = label[:,:,i] - mask
        tmp[tmp<0] = 0
        label[:,:,i] = tmp
        img[:,:,i] = img[:,:,i] + label[:,:,i]
    '''
    #==================================================
    image=cv2.imread(file_path, cv2.IMREAD_COLOR)
    
    # preprocess
    # norm_image = base_transform(image, (320, 320), MEANS)
    norm_image = cv2.resize(image, (320, 320)).astype(np.float32)
    norm_image -= MEANS
    norm_image = norm_image.astype(np.float32)
    norm_image = torch.from_numpy(norm_image).permute(2, 0, 1)

    # forward
    input_var = Variable(norm_image.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        input_var = input_var.cuda()
    detection = net(input_var)


    # scale each detection back up to the image,
    # scale = (width, height, width, height)
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)
    threshold = 0.5
    num_top = detection.size(2)
    colors = (plt.cm.hsv(np.linspace(0, 1, num_classes)) * 255).tolist()
    for i in range(1, num_classes):
        for j in range(num_top):
            score = detection[0, i, j, 0]
            if score < threshold:
                continue
            label_name = class_names[i]
            display_txt = '%s: %.2f' % (label_name, score)
            pts = (detection[0, i, j, 1:] * scale).cpu().numpy().astype(np.int32)
            pts = tuple(pts)
            cv2.rectangle(image, pts[:2], pts[2:], colors[i], 4)
            cv2.putText(image, display_txt,
                pts[:2],
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2.5, colors[i])
    # pdb.set_trace()
    name, ext = os.path.split(file_path)
    cv2.imwrite(name + '_result' + ext, image)
    #=================================
    cv2.imwrite("./output/save.png", image)
    img = Image.open("./output/save.png")
    img_bottom(img)
    



file = ttk.Button(frm_top, text = "Open File", command = openfile)
file.grid(row = 0, column = 0, padx = 5, pady = 5)

hist = ttk.Button(frm_top, text = "Detection", command = detection)
hist.grid(row = 0, column = 1, padx = 5, pady = 5)


win.mainloop()