import os
import csv
import random
import pathlib
import argparse
from tokenize import Double
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd

import torch
import torchvision
from torch.nn import functional as F
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import D2NN_classification


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class data_loader(Dataset):
    def __init__(self, root, split, transform):
        self.transform = transform
        file_path = root+"\\"+split+"\\input\\data.xlsx"
        self.data = np.array(pd.read_excel(file_path))
        size = self.data.shape[1]
        file_path = root+"\\"+split+"\\input\\data_1.xlsx"
        self.data_1 = np.array(pd.read_excel(file_path))
        file_path = root+"\\"+split+"\\gt\\label.xlsx"
        self.label = np.array(pd.read_excel(file_path)).squeeze(0)
        print(len(self.label),size)
        assert size == len(self.label), "mismatched label' and img' length"
    
    def __getitem__(self, index):
        data = self.data[:,index] + self.data_1[:,index]
        lbl = self.label[index]

        
        return data, lbl
    
    def __len__(self):
        return len(self.label)
    
#######################//  Tamsform  //#####################################
    
trainTransform = transforms.Compose([
    transforms.ToTensor()
])
testTransform = transforms.Compose([
    transforms.ToTensor()
])

#######################// Data loader //##############################################



def main(args):

    #################// Data loader //#############################3
    train_set = data_loader(root = args.root_path, split = "train",
                            transform = trainTransform)
    train_dataloader = DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True )
    test_set = data_loader(root = args.root_path, split = "test",
                            transform = trainTransform)
    val_dataloader = DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = True)
    

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    model = D2NN_classification.Net()
    model.cuda()

    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))
        print('Model : "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
    else:
        if os.path.exists(args.result_record_path):
            os.remove(args.result_record_path)
        else:
            with open(args.result_record_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ['Epoch', 'Train_Loss', "Train_Acc", 'Val_Loss', "Val_Acc", "LR"])
                
    criterion = torch.nn.MSELoss(reduction = "sum").cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)


    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.num_epochs):

        log = [epoch]
        model.train()

        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0

        tk0 = tqdm(train_dataloader, ncols=100, total=int(len(train_dataloader)))
        for train_iter, train_data_batch in enumerate(tk0):

            train_images = train_data_batch[0].cuda()
            train_labels = train_data_batch[1].cuda()
            

            train_labels = F.one_hot(train_labels, num_classes=2)
            train_images = torch.squeeze(train_images).cuda()
            #train_images = torch.squeeze(torch.cat((train_images.unsqueeze(-1),
            #                                        torch.zeros_like(train_images.unsqueeze(-1))), dim=-1), dim=1)
            
            train_outputs = model(train_images.float())
            print(train_outputs)
            print(train_labels)

            train_loss_ = criterion(train_outputs, train_labels.float())
            print(train_loss_)
            train_counter_ = torch.eq(torch.argmax(train_labels, dim = 1), torch.argmax(train_outputs, dim = 1)).float().sum()

            optimizer.zero_grad()
            train_loss_.backward()
            optimizer.step()

            train_len += len(train_labels)
            train_running_loss += train_loss_.item()
            train_running_counter += train_counter_

            train_loss = train_running_loss / train_len
            train_accuracy = train_running_counter / train_len

        log.append(train_loss)
        log.append(train_accuracy)

        with torch.no_grad():

            model.eval()

            val_len = 0.0
            val_running_counter = 0.0
            val_running_loss = 0.0

            tk1 = tqdm(val_dataloader, ncols = 100, total = int(len(val_dataloader)))
            for val_iter, val_data_batch in enumerate(tk1):

                val_images = val_data_batch[0].cuda()
                val_labels = val_data_batch[1].cuda()
                val_labels = F.one_hot(val_labels, num_classes=2)


                val_images = torch.squeeze(train_images).cuda()
                #val_images = torch.squeeze(torch.cat((val_images.unsqueeze(-1),
                #                                        torch.zeros_like(val_images.unsqueeze(-1))), dim=-1), dim=1)

                val_outputs = model(val_images.float())

                val_loss_ = criterion(val_outputs, val_labels.float())
                val_counter_ = torch.eq(torch.argmax(val_labels, dim=1), torch.argmax(val_outputs, dim=1)).float().sum()

                val_len += len(val_labels)
                val_running_loss += val_loss_.item()
                val_running_counter += val_counter_

                val_loss = val_running_loss / val_len
                val_accuracy = val_running_counter / val_len

                tk1.set_description_str('Epoch {}/{} : Validating'.format(epoch, args.start_epoch + 1 + args.num_epochs - 1))
                tk1.set_postfix({'Val_Loss': '{:.5f}'.format(val_loss), 'Val_Accuarcy': '{:.5f}'.format(val_accuracy)})

            log.append(val_loss)
            log.append(val_accuracy)

        torch.save(model.state_dict(), (args.model_save_path + str(epoch) + args.model_name))
        print('Model : "' + args.model_save_path + str(epoch) + args.model_name + '" saved.')

        with open(args.result_record_path, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.96)
    parser.add_argument('--whether-load-model', type=bool, default=False, help="whether need to continus")
    parser.add_argument('--start-epoch', type=int, default=0, help='which epoch')
    
    parser.add_argument('--root_path', type=str, default=r"C:\Users\Dennis\Desktop\SHW\ANN\data")
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="saving path")

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)



