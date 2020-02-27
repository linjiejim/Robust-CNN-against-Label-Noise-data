import torch
import numpy as np
from model import CnnModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import utils
from options import CnnModelOptions
opt = CnnModelOptions().parse()
import os

def validate(model, X_val, y_val):

	# valiadate the model
	model.eval()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	inputs, targets = torch.autograd.Variable(torch.Tensor(X_val)).float().to(device), \
					  torch.autograd.Variable(torch.Tensor(y_val)).float().to(device)
	y_pred = model.predict(inputs).cpu().detach().numpy()
	y_pred = np.argmax(y_pred, axis=1)

	accuracy = accuracy_score(y_val, y_pred)
	precision = precision_score(y_val, y_pred, average='weighted')
	recall = recall_score(y_val, y_pred, average='weighted')

	return accuracy, precision, recall

def train(trainloader, model, X_val, y_val):

	# train the model
	model.train()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	optimizer = model.optimizer
	epoch = opt.epoch
	criterion = utils.get_loss(opt.loss_name, opt)

	for i in range(epoch):
		average_loss = 0
		count = 0
		
		for batch_idx, (inputs, targets_onehot) in enumerate(trainloader):
			inputs, targets_onehot = torch.autograd.Variable(inputs).float().to(device), torch.autograd.Variable(targets_onehot).long().to(device)

			# make predictions
			outputs = model(inputs)

			# compute loss
			loss = criterion(outputs, targets_onehot)

			average_loss += loss.item()
			count += 1

			# update networks
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		average_loss /= count

		if opt.print_loss:
			accuracy, _, _ = validate(model, X_val, y_val)
			print("Epoch = {}, loss={}, accuracy = {}".format(i, average_loss, accuracy))
			model.train()

	if opt.save_model:
		print('saving the model at the end of epoch %d' % (epoch))
		model.save_network(epoch)

if __name__ == '__main__':

	opt.pretrained_model = None
	num_classes = opt.num_classes

	# load the dataset
	X_train, y_train, y_train_onehot, X_test, y_test = utils.read_dataset(
		os.path.join(opt.dataset_root, opt.dataset_name))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	X_train, y_train, y_train_onehot = utils.augment(X_train, y_train, y_train_onehot)

	n_val = int(len(X_train) * opt.val_split_rate)
	X_train, y_train, y_train_onehot, X_val, y_val, y_val_onehot = \
		X_train[n_val:], y_train[n_val:], y_train_onehot[n_val:], \
		X_train[:n_val], y_train[:n_val], y_train_onehot[:n_val]

	dataloader = torch.utils.data.DataLoader(
		list(zip(X_train, y_train_onehot)),
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=4
	)

	if opt.loss_name == "mpe" and opt.trans_matrix is None:

		# compute transition matrix
		trans_matrices1 = []
		trans_matrices2 = []

		for i in range(opt.num_trained_model):

			pretrained_model = CnnModel(opt).to(device)

			# load the pretrained model
			opt.pretrained_model_path = opt.pretrained_model_path_format.format(i, i)
			assert opt.pretrained_model_path != ""
			pretrained_model.load(opt.pretrained_model_path)

			# take model to eval status
			pretrained_model.eval()
			accuracy, precision, recall = validate(pretrained_model, X_test, y_test)
			print("model_accuracy = {}".format(accuracy))
			matrix1 = utils.compute_T(pretrained_model, X_train, opt.num_classes)
			matrix2 = utils.compute_T2(pretrained_model, X_train, opt.num_classes)
			trans_matrices1.append(matrix1)
			trans_matrices2.append(matrix2)
			print(matrix1)
			print(matrix2)


		if (len(opt.log_file_path) > 0):
			utils.log_to_file("Model Name: {} ".format(opt.name), opt.log_file_path)
			utils.log_to_file("Transition Matrix 1", opt.log_file_path)
			utils.log_to_file("{}".format(np.mean(trans_matrices1, axis=0)), opt.log_file_path)
			utils.log_to_file("Transition Matrix 2", opt.log_file_path)
			utils.log_to_file("{}".format(np.mean(trans_matrices2, axis=0)), opt.log_file_path)
			utils.log_to_file("-----------------------", opt.log_file_path)
		else:
			print("Model Name: {} ".format(opt.name))
			print("Transition Matrix 1")
			print("{}".format(np.mean(trans_matrices1, axis=0)))
			print("Transition Matrix 2")
			print("{}".format(np.mean(trans_matrices2, axis=0)))
			print("-----------------------")

	elif opt.is_training:

		accs, pres, recs = [], [], []

		for i in range(opt.num_trained_model):

			model = CnnModel(opt).to(device)

			# train the model
			print("Training start!")
			train(dataloader, model, X_val, y_val)

			# test the model with testing data
			print("Testing start!")
			accuracy, precision, recall = validate(model, X_test, y_test)
			accs.append(accuracy)
			pres.append(precision)
			recs.append(recall)

		if (len(opt.log_file_path) > 0):
			utils.log_to_file("Model Name: {} ".format(opt.name), opt.log_file_path)
			utils.log_to_file("Accuracy Mean: {} Std: {}".format(np.mean(accs), np.std(accs)), opt.log_file_path)
			utils.log_to_file("Precision Mean: {} Std: {}".format(np.mean(pres), np.std(pres)), opt.log_file_path)
			utils.log_to_file("Recall: Mean: {} Std: {}".format(np.mean(recs), np.std(recs)), opt.log_file_path)
			utils.log_to_file("-----------------------", opt.log_file_path)

	else:

		model = CnnModel(opt).to(device)

		# load trained model
		model.load(opt.pretrained_model_path)

		# test the model with testing data
		print("Testing start!")
		validate(model, X_test, y_test)

