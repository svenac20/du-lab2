import torch
import numpy as np
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from pathlib import Path


class ConvolutionalModel(nn.Module):
	def __init__(self, in_channels, conv1_width, max_pool1_kernel_size, conv2_out, max_pool2_kernel_size, fc1_width,
				 class_count):
		super(ConvolutionalModel, self).__init__()
		# ulaz 28x28, izlaz 28x28x16
		self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
		# ulaz 28x28x16, izlaz 14x14x16
		self.maxPooling1 = nn.MaxPool2d(kernel_size=max_pool1_kernel_size)

		# ulaz 14x14x16, izlaz 14x14x32
		self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_out, kernel_size=5, stride=1, padding=2,
							   bias=True)
		# ulaz 14x14x32, izlaz 7x7x32
		self.maxPooling2 = nn.MaxPool2d(kernel_size=max_pool2_kernel_size)
		# potpuno povezani slojevi
		self.fc1 = nn.Linear(in_features=conv2_out * 7 * 7, out_features=fc1_width, bias=True)
		self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

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
		h = self.maxPooling1(h)
		h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)

		h = self.conv2(h)
		h = self.maxPooling2(h)
		h = torch.relu(h)

		h = h.view(h.shape[0], -1)

		h = self.fc1(h)
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

	def train(self, train, validation, number_of_epochs=8, weight_decay=1e-3, delta=0.1):
		optimizer = torch.optim.SGD(params=self.parameters(), weight_decay=weight_decay, lr=delta)

		writer = SummaryWriter()
		loss_fn = torch.nn.CrossEntropyLoss()

		losses_for_epoch = []
		for i in range(1, number_of_epochs + 1):
			for x, y in train:
				out = self.forward(x)

				loss = loss_fn(out, y)
				loss.backward()

				optimizer.step()
				optimizer.zero_grad()

			validation_loss = self.get_loss_for_validation_loader(validation)

			writer.add_scalar("Validation loss", round(validation_loss, 5), i)
			print(f"Loss for epoch number - {i} : {validation_loss}")
			losses_for_epoch.append(validation_loss)

		writer.close()

		plt.plot(range(1, number_of_epochs + 1), losses_for_epoch)
		plt.title("Validation loss through epochs")
		plt.show()

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'


def get_MNIST_dataset(batch_size):
	train_dataset = MNIST(root=DATA_DIR.__str__(), transform=torchvision.transforms.ToTensor(), download=True)
	test_dataset = MNIST(root=DATA_DIR.__str__(), transform=torchvision.transforms.ToTensor(), train=False,
						 download=True)

	train_set, validation_set = torch.utils.data.random_split(train_dataset, [55000, 5000])

	train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

	return train_loader, validation_loader, test_loader


if __name__ == '__main__':
	batch_size = 50

	train_loader, validation_loader, test_loader = get_MNIST_dataset(batch_size)

	conv_model = ConvolutionalModel(in_channels=1, conv1_width=16, max_pool1_kernel_size=2, conv2_out=32,
									max_pool2_kernel_size=2, fc1_width=512, class_count=10)
	conv_model.train(train_loader, validation_loader)
