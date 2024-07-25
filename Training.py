import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os 
import optuna
import copy
import ResNet
import DataSetLoader
from tqdm import tqdm
import nni
import json

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
	"""
	Arguments:
	dataset - data set used when training the model
	TODO batch size - ???
    learning rate - initial learning rate of the networks
	momentum - SGD momentum
	weight decay - SGD weight decay
	teacher - teacher name
	student - student name
	teacher checkpoint - pretrained teacher model
	cuda - cuda
	TODO dataset - make this link with datasetloader.py
	"""
	parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
	parser.add_argument('--epochs', default=10, type=int,  help='number of total epochs to run')
	#parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be either cifar10 or cifar100') #TODO make this compatable with my dataset
	parser.add_argument('--batch-size', default=8, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	parser.add_argument('--teacher', default='', type=str, help='shape of teacher model')
	parser.add_argument('--student', '--model', default='4', type=str, help='shape of student model')
	parser.add_argument('--teacher-checkpoint', default='', type=str, help='optinal pretrained checkpoint for teacher')
	parser.add_argument('--cuda', default='', type=str, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	args = parser.parse_args()
	return args


class TrainManager(object):
	def __init__(self, student, teacher=None, train_loader=None, test_loader=None, train_config={}, label_map_file=None):
		self.student = student
		self.teacher = teacher
		self.label_map_file = label_map_file
		self.have_teacher = bool(self.teacher)
		self.device = train_config['device']
		self.name = train_config['name']
		self.optimizer = optim.SGD(self.student.parameters(),
								   lr=train_config['learning_rate'],
								   momentum=train_config['momentum'],
								   weight_decay=train_config['weight_decay'])
		if self.have_teacher:
			self.teacher.eval()
			self.teacher.train(mode=False)
			
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.config = train_config
	
	def train(self):
		lambda_ = self.config['lambda_student']
		T = self.config['T_student']
		epochs = self.config['epochs']
		trial_id = self.config['trial_id']
		
		max_val_acc = 0
		best_acc = 0
		criterion = nn.CrossEntropyLoss()
		with open(self.label_map_file, 'r') as file:
			label_mapping = json.load(file)

		for epoch in tqdm(range(epochs),desc="EPOCHS"):
			self.student.train()
			self.adjust_learning_rate(self.optimizer, epoch)
			loss = 0
			
			for batch_idx, (inputs, labels) in enumerate(tqdm(self.train_loader,desc="Training")):
				labels = [label_mapping[l] for l in labels] 
				inputs = inputs.clone().detach().to(self.device) #inputs already tensor
				labels = torch.tensor(labels, dtype=torch.long).to(self.device)

				self.optimizer.zero_grad()
				output = self.student(inputs)

				loss_SL = criterion(output, labels) 
				loss = loss_SL
				
				if self.have_teacher:
					teacher_outputs = self.teacher(inputs)
					# Knowledge Distillation Loss
					loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1), F.softmax(teacher_outputs / T, dim=1))

					loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
					
				loss.backward()
				self.optimizer.step()
			
			#print("epoch {}/{}".format(epoch, epochs))
			val_acc = self.validate(step=epoch)
			if val_acc > best_acc:
				best_acc = val_acc
				self.save(epoch, name='{}_{}_best.pth.tar'.format(self.name, trial_id))
		
		return best_acc
	
	def validate(self, step=0):
		self.student.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			acc = 0
			for images, labels in self.test_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				outputs = self.student(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			# self.accuracy_history.append(acc)
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
	config = nni.get_next_parameter()
	default_seed = 42  # Default seed if not provided by NNI
	seed = config.get('seed', default_seed)
	torch.manual_seed(seed)
	trial_id = os.environ.get('NNI_TRIAL_JOB_ID')
	#dataset = args.dataset
	num_classes = 97#TODO method of determing class size from dataset
	teacher_model = None
	student_model = ResNet.ResNet_Model(ResNet.Bottleneck, model_shapes.get(args.student), num_classes=num_classes) 
	train_config = {
	'epochs': args.epochs,
	'learning_rate': args.learning_rate,
	'momentum': args.momentum,
	'weight_decay': args.weight_decay,
	'device': 'cuda' if args.cuda else 'cpu',
	'trial_id': trial_id,
	'T_student': config.get('T_student'),
	'lambda_student': config.get('lambda_student'),	
	}

	if args.teacher:	
		teacher_model = ResNet.ResNet_Model(ResNet.Bottleneck, model_shapes.get(args.teacher), num_classes=num_classes) # TODO pass correct arguments
		if args.teacher_checkpoint:
			print("---------- Loading Teacher -------")
			teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
		else:
			print("---------- Training Teacher -------")
			train_loader,test_loader,label_file_name = DataSetLoader.load_dataset(batch_size=args.batch_size)
			teacher_train_config = copy.deepcopy(train_config)
			teacher_name = '{}_{}_best.pth.tar'.format(args.teacher, trial_id)
			teacher_train_config['name'] = args.teacher
			teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config,label_map_file=label_file_name)
			teacher_trainer.train()
			teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))

	print("---------- Training Student -------")
	student_train_config = copy.deepcopy(train_config)
	train_loader,test_loader,label_file_name = DataSetLoader.load_dataset(batch_size=args.batch_size)
	student_train_config['name'] = args.student
	student_trainer = TrainManager(student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config,label_map_file=label_file_name)
	best_student_acc = student_trainer.train()
	nni.report_final_result(best_student_acc)


	
