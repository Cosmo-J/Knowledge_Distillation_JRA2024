import csv
import os
import argparse
import datetime
import settings
import torch

class Saver:
    def __init__(self, csvName: str = None, col: list = ['Col1', 'Col2'],datetime = False):
        if csvName is None:
            raise ValueError("A valid CSV name must be provided.")
        self.name = csvName
        self.outfile = csvName + '.csv'
        self.outfilepath = settings.MODELS_PATH + f"{csvName}/"
        self.columns = col
        self.length = len(col)
        if (datetime):
            self.columns.append('Time')


        if not os.path.exists(self.outfilepath):
            os.makedirs(self.outfilepath)

        if not os.path.isfile(self.outfile):
            # Create the CSV file and write the header
            with open(self.outfilepath + self.outfile, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)

    def SaveCsv(self, data):
        if len(data) != self.length:
            raise ValueError("Data length does not match the number of columns.")
        if not isinstance(data, list):
            raise ValueError("Data must be provided as a list.")
        data.append(datetime.datetime.now())
        with open(self.outfilepath + self.outfile, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def SaveModel(self,student_state,optimizer_state,epoch):
        saveName = '{}_epoch{}.pth.tar'.format(self.name, epoch)
        savePath = self.outfilepath+ saveName
        torch.save({'epoch': epoch,'model_state_dict': student_state,'optimizer_state_dict': optimizer_state,},savePath)


def parse_arguments(): 

    # type how comprehensive the test is against the data set
    # model.tar used
    # print results
    # save results to csv
    # info.txt

    parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code'); 
    parser.add_argument('--model', default='', type=str, help='Path to model being tested')
    parser.add_argument('--cuda', default='', type=str, help='whether or not use cuda')
    parser.add_argument('--display',default='True',type=str, help='whether or not to print results to terminal')
    parser.add_argument('--save', default='',type=str,help ='whether or not to save results to a csv')
    args = parser.parse_args()
    return args