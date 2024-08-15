import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os 
import copy
import ResNet
import DataSetLoader
from tqdm import tqdm
import json
import settings
import TestResults
import datetime

def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model

def parse_arguments(): 
	parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
	parser.add_argument('--epochs', default=10, type=int,  help='number of total epochs to run')
	parser.add_argument('--batch-size', default=16, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	parser.add_argument('--teacher', default='', type=str, help='shape of teacher model')
	parser.add_argument('--student', '--model', default='4', type=str, help='shape of student model')
	parser.add_argument('--student-checkpoint', default='', type=str, help='option to continuation of training a given model')
	parser.add_argument('--teacher-checkpoint', default='', type=str, help='optional pretrained checkpoint for teacher')
	parser.add_argument('--cuda', default='', type=str, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	parser.add_argument('--dataset-size', default='1', type=float, help='percentage of dataset used')
	parser.add_argument('--data-prefetch', default='1', type=int, help='how many batches are prefetched by the data loader')
	parser.add_argument('--loader-workers', default='1', type=int, help='how many workers/threads are used to download data by the data loader')

	args = parser.parse_args()
	return args


class TrainManager(object):
	def __init__(self, student, teacher=None, train_loader=None, test_loader=None, train_config={}, label_map_file=None,resulter=None):
		self.device = train_config['device']
		self.student = student.to(self.device)
		self.teacher = teacher.to(self.device) if teacher else None
		self.resulter = resulter

		self.label_map_file = label_map_file
		self.name = train_config['name']
		self.optimizer = optim.SGD(self.student.parameters(),
								   lr=train_config['learning_rate'],
								   momentum=train_config['momentum'],
								   weight_decay=train_config['weight_decay'])
		
		self.have_teacher = bool(self.teacher)
		if self.have_teacher:
			self.teacher.eval()
			self.teacher.train(mode=False)
			
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.config = train_config
	
	def train(self):
		print("Student model device:", next(self.student.parameters()).device)
		lambda_ = self.config['lambda_student']
		T = self.config['T_student']
		epochs = self.config['epochs']
		trial_id = self.config['trial_id']
		
		max_val_acc = 0
		best_acc = 0
		criterion = nn.CrossEntropyLoss()
		with open(self.label_map_file, 'r') as file:
			self.label_mapping = json.load(file)

		for epoch in tqdm(range(epochs),desc="EPOCHS"):
			self.student.train()
			self.adjust_learning_rate(self.optimizer, epoch)
			loss = 0
			for batch_idx, (inputs, labels) in enumerate(tqdm(self.train_loader,desc="Training")):

				labels = [self.label_mapping[l] for l in labels] 

				inputs = inputs.clone().detach().to(self.device) #inputs already tensor
				labels = torch.tensor(labels, dtype=torch.long).to(self.device)

				self.optimizer.zero_grad()
				output = self.student(inputs)

				loss_SL = criterion(output, labels) 
				loss = loss_SL
				
				if self.have_teacher:
					teacher_outputs = self.teacher(inputs)
					loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1), F.softmax(teacher_outputs / T, dim=1))
					loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
				loss.backward()

				self.optimizer.step()
			
			val_acc = self.validate(step=epoch)

			if(self.resulter != None):
				self.resulter.SaveCsv([epoch,val_acc])


			if val_acc > best_acc:
				best_acc = val_acc
				self.save(epoch)
		
		return best_acc
	
	def validate(self, step=0):
		self.student.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			acc = 0
			for inputs, slabels in self.test_loader:
				labels = [self.label_mapping[l] for l in slabels] 
				print(f"slabels:{slabels}")
				inputs = inputs.clone().detach().to(self.device)
				labels = torch.tensor(labels, dtype=torch.long).to(self.device)
				outputs = self.student(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			acc = 100 * correct / total
			
			print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
			return acc
	
	def save(self, epoch, name=None):
		trial_id = self.config['trial_id']
		if name is None:
			torch.save({
				'epoch': epoch,
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
			}, '{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
		else:
			torch.save({
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'epoch': epoch,
			}, name)
	
	def adjust_learning_rate(self, optimizer, epoch):
		epochs = self.config['epochs']
		#models_are_plane = self.config['is_plane']
		
		# depending on dataset
		#if models_are_plane:
		#	lr = 0.01
		#else:
		if epoch < int(epoch/2.0):
			lr = 0.1
		elif epoch < int(epochs*3/4.0):
			lr = 0.1 * 0.1
		else:
			lr = 0.1 * 0.01
		
		# update optimizer's learning rate
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


model_shapes = {
'1':[1,1,1],
'2':[2,2,2],
'3':[3,3,3],
'4':[4,4,4],
'5':[5,5,5],
'7':[7,7,7],
'9':[9,9,9],
'18':[18,18,18],
'200':[200,200,200],
}


if __name__ == "__main__":
	args = parse_arguments()
	print(args)
	default_seed = 26  
	torch.manual_seed(default_seed)
	trial_id = "test"
	num_classes = settings.NUM_CLASSES

	train_config = {
		'name': args.student + trial_id,
		'epochs': args.epochs,
		'learning_rate': args.learning_rate,
		'momentum': args.momentum,
		'weight_decay': args.weight_decay,
		'device': 'cuda' if args.cuda.lower() =='true' and torch.cuda.is_available() else 'cpu',
		'trial_id': trial_id,
		'T_student': 1,
		'lambda_student': 0.05,	
	}
	teacher_model = None
	student_model = ResNet.ResNet_Model(ResNet.Bottleneck, model_shapes.get(args.student), num_classes=num_classes,device=train_config['device']) 

	if args.teacher:	
		teacher_model = ResNet.ResNet_Model(ResNet.Bottleneck, model_shapes.get(args.teacher), num_classes=num_classes,device=train_config['device'])
		if args.teacher_checkpoint:
			print("---------- Loading Teacher -------")
			teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
		else:
			print("---------- Training Teacher -------")
			train_loader,test_loader,label_file_name = DataSetLoader.load_dataset(batch_size=args.batch_size,dataset_percentage=args.dataset_size,prefetch_factor=args.data_prefetch,num_workers=args.loader_workers)
			DataSetLoader.plot_class_distribution(train_loader=train_loader)

			teacher_train_config = copy.deepcopy(train_config)
			teacher_name = '{}_{}_best.pth.tar'.format(args.teacher, trial_id)
			teacher_train_config['name'] = args.teacher
			teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config,label_map_file=label_file_name)
			teacher_trainer.train()
			teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))


	
	if args.student_checkpoint:
		#name = os.path.basename(args.student_checkpoint)
		print("---------- Coninuing Training -------")
		student_checkpoint_path = args.student_checkpoint
		student_model = load_checkpoint(student_model,student_checkpoint_path)
	

	print("---------- Training Student -------")
	student_train_config = copy.deepcopy(train_config)
	train_loader,test_loader,label_file_name = DataSetLoader.load_dataset(batch_size=args.batch_size,dataset_percentage=args.dataset_size,prefetch_factor=args.data_prefetch,num_workers=args.loader_workers)
	student_train_config['name'] = args.student
	name = student_train_config['name']
	resulter = TestResults.Saver(name,['Epoch','Accuracy'],datetime=True)
	student_trainer = TrainManager(student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config,label_map_file=label_file_name,resulter=resulter)
	best_student_acc = student_trainer.train()


	
