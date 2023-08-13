import os
import gdown
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optims

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
# 
from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    print("image_name--",image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def get_icc_profile(im1):
    # im1 = Image.open(image_path)
    raw_icc_profile = im1.info.get("icc_profile")
    return raw_icc_profile

def save_transparent(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    mask = Image.fromarray(predict_np*255).convert("L")
    img_name = image_name.split(os.sep)[-1]
    image = Image.open(image_name)
    icc_profile = get_icc_profile(image) 
    mask = mask.resize((image.size),resample=Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    image.putalpha(mask)
    image.save(os.path.join(d_dir,imidx+'.png'),icc_profile=icc_profile)

def process_saliency(image_dir=None,prediction_dir=None):

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    transforms_test= transforms.Compose([
        transforms.PILToTensor(),RescaleT(320),ToTensorLab(flag=0)
    ])


    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images') if image_dir is None else image_dir
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep) if prediction_dir is None else prediction_dir
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    if not os.path.exists(model_dir):
        os.makedirs('./saved_models/u2net', exist_ok=True)
        gdown.download('https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
    './saved_models/u2net/u2net.pth',
    quiet=False)


    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    ## --------- 2. dataloader ---------
    ## 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        # inputs_test = Image.open(data_test)
        # inputs_test = inputs_test.resize((320,320))
        # inputs_test = inputs_test.type(torch.FloatTensor)
        # inputs_test = transforms_test(inputs_test)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        # save_output(img_name_list[i_test],pred,prediction_dir)
        save_transparent(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

    # for i_test, data_test in enumerate(test_salobj_dataloader):

    #     print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

    #     inputs_test = data_test['image']
    #     inputs_test = inputs_test.type(torch.FloatTensor)

    #     if torch.cuda.is_available():
    #         inputs_test = Variable(inputs_test.cuda())
    #     else:
    #         inputs_test = Variable(inputs_test)

    #     d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    #     # normalization
    #     pred = d1[:,0,:,:]
    #     pred = normPRED(pred)

    #     # save results to test_results folder
    #     if not os.path.exists(prediction_dir):
    #         os.makedirs(prediction_dir, exist_ok=True)
    #     # save_output(img_name_list[i_test],pred,prediction_dir)
    #     save_transparent(img_name_list[i_test],pred,prediction_dir)

    #     del d1,d2,d3,d4,d5,d6,d7

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
