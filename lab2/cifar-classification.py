import torch
import numpy as np
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path


class ConvolutionalModel(nn.Module):
	def __init__(self, in_channels, conv1_width, max_pool1_kernel_size, conv2_out, max_pool2_kernel_size, fc1_width, fc2_width,
				 class_count):
		super(ConvolutionalModel, self).__init__()
		# ulaz 32x32x3, izlaz 32x32x16
		self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
		# ulaz 32x32x16, izlaz 15x15x16
		self.maxPooling1 = nn.MaxPool2d(kernel_size=max_pool1_kernel_size, stride=2)

		# ulaz 15x15x16, izlaz 15x15x32
		self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_out, kernel_size=5, stride=1, padding=2,
							   bias=True)
		# ulaz 15x15x32, izlaz 7x7x32
		self.maxPooling2 = nn.MaxPool2d(kernel_size=max_pool2_kernel_size, stride=2)
		# potpuno povezani slojevi
		self.fc1 = nn.Linear(in_features=conv2_out * 7 * 7, out_features=fc1_width, bias=True)
		self.fc2 = nn.Linear(in_features=fc1_width, out_features=fc2_width, bias=True)
		self.fc_logits = nn.Linear(fc2_width, class_count, bias=True)

		# parametri su već inicijalizirani pozivima Conv2d i Linear
		# ali možemo ih drugačije inicijalizirati
		self.reset_parameters()

	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear) and m is not self.fc_logits:
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
		self.fc_logits.reset_parameters()

	def forward(self, x):
		h = self.conv1(x)
		h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
		h = self.maxPooling1(h)

		h = self.conv2(h)
		h = torch.relu(h)
		h = self.maxPooling2(h)

		h = h.view(h.shape[0], -1)

		h = self.fc1(h)
		h = torch.relu(h)
		h = self.fc2(h)
		h = torch.relu(h)
		logits = self.fc_logits(h)

		return logits

	def get_loss_for_validation_loader(self, validation_loader):
		loss_fn = torch.nn.CrossEntropyLoss()

		loss_per_batch = torch.zeros(len(validation_loader))
		for index, (x, y) in enumerate(validation_loader):
			out = self.forward(x)
			loss = loss_fn(out, y)

			loss_per_batch[index] = loss

		return torch.mean(loss_per_batch).item()

	def get_classification(self, out):
		return torch.argmax(out, dim=1)

	def evaluate(self, train, validation):
		accuracy_train_validation = []
		loss_train_validation = []

		with torch.no_grad():
			loss_fn = torch.nn.CrossEntropyLoss()
			for i in range(2):
				dataset = train if i == 0 else validation

				y_true_labels = None
				y_classification_labels = None
				losses = []

				for x, y in dataset:
					out = self.forward(x)
					loss = loss_fn(out, y)

					losses.append(loss)
					classification = self.get_classification(out)

					if y_true_labels is None:
						y_true_labels = y
					else:
						y_true_labels = torch.vstack((y_true_labels, y))

					if y_classification_labels is None:
						y_classification_labels = classification
					else:
						y_classification_labels = torch.vstack((y_classification_labels, classification))

				matrix = confusion_matrix(y_true_labels.reshape(-1), y_classification_labels.reshape(-1))
				accuracy_train_validation.append(accuracy_score(y_true_labels.reshape(-1), y_classification_labels.reshape(-1)))
				loss_train_validation.append(np.mean(losses))

		return accuracy_train_validation, loss_train_validation


	def train(self, train, validation, number_of_epochs=10, weight_decay=1e-3, delta=0.01):
		optimizer = torch.optim.SGD(params=self.parameters(), weight_decay=weight_decay, lr=delta)

		exponentialLR = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

		accuracy_train = []
		accuracy_validation = []

		loss_train = []
		loss_validation = []

		learning_rate = []

		loss_fn = torch.nn.CrossEntropyLoss()

		losses_for_epoch = []
		for i in range(1, number_of_epochs + 1):
			for x, y in train:
				out = self.forward(x)

				loss = loss_fn(out, y)
				loss.backward()

				optimizer.step()
				optimizer.zero_grad()

			exponentialLR.step()
			self.evaluate(train, validation)

			accuracy_train_validation, loss_train_validation = self.evaluate(train, validation)
			print(f"Accuracy on train after epoch {i}: {accuracy_train_validation[0]}")
			print(f"Average loss on train after epoch {i}: {loss_train_validation[0]}")
			learning_rate.append(exponentialLR.get_last_lr())

			print(f"Accuracy on validation after epoch {i}: {accuracy_train_validation[1]}")
			print(f"Average loss on validation after epoch {i}: {loss_train_validation[1]}")

			accuracy_train.append(accuracy_train_validation[0])
			loss_train.append(loss_train_validation[0])

			accuracy_validation.append(accuracy_train_validation[1])
			loss_validation.append(loss_train_validation[1])

		plt.title("Accuracy on validation and train set through epochs")
		plt.plot(range(1, number_of_epochs + 1), accuracy_train)
		plt.plot(range(1, number_of_epochs + 1), accuracy_validation)
		plt.legend(["Train accuracy", "Validation accuracy"])
		plt.show()

		plt.title("Average loss on validation and train set through epochs")
		plt.plot(range(1, number_of_epochs + 1), loss_train)
		plt.plot(range(1, number_of_epochs + 1), loss_validation)
		plt.legend(["Train loss", "Validation loss"])
		plt.show()

		plt.title("Learning rate through epochs")
		plt.plot(range(1, number_of_epochs + 1), learning_rate)
		plt.show()

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'


def get_CIFAR_dataset(batch_size):
	transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
													  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
																					   (0.2023, 0.1994, 0.2010))])

	train_dataset = CIFAR10(root=DATA_DIR.__str__(), transform=transform_train, download=True)
	test_dataset = CIFAR10(root=DATA_DIR.__str__(), transform=transform_train, train=False, download=True)

	train_set, validation_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

	train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

	return train_loader, validation_loader, test_loader


if __name__ == '__main__':
	batch_size = 50

	train_loader, validation_loader, test_loader = get_CIFAR_dataset(batch_size)

	conv_model = ConvolutionalModel(in_channels=3, conv1_width=16, max_pool1_kernel_size=3, conv2_out=32,
									max_pool2_kernel_size=3, fc1_width=256, fc2_width=128, class_count=10)
	conv_model.train(train_loader, validation_loader)
