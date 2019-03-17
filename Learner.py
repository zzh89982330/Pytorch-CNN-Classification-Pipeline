from torch.utils.data import DataLoader
from torch.nn import Module
import torch

class Learner:
    def __init__(self, train_loader:DataLoader, model:Module, epochs, loss_fn, optimizer, prt_loss_ite, test_loader, ite_after_fn, epoch_after_fn):
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prt_loss_ite = prt_loss_ite
        self.test_loader = test_loader
        self.ite_after_fn = ite_after_fn
        self.epoch_after_fn = epoch_after_fn
        self.ite_per_epoch = len(train_loader)

    def learn(self):
        ite_count = 0
        for epoch in range(self.epochs):
            for data, label in self.train_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                ite_count += 1
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.loss_fn(out, label) #type:torch.Tensor
                loss.backward()
                self.optimizer.step()

                if self.ite_after_fn:
                    for func in self.ite_after_fn:
                        func(self, ite_count)

                if ite_count % self.prt_loss_ite == 0:
                    print("<ite {}[{}/{}]: model loss: {}>".format(ite_count, ite_count, self.ite_per_epoch, loss.cpu().detach().item()))

            if self.epoch_after_fn:
                for func in self.epoch_after_fn:
                    func(self, epoch)

    def evl_model(self, back_to_train = True):
        correct = 0
        total = 0
        
        torch.cuda.empty_cache()

        for para in self.model.parameters():
            para.requires_grad = False

        for images, labels in self.test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        torch.cuda.empty_cache()
        
        if back_to_train:
            for para in self.model.parameters():
                para.requires_grad = True

        print("Test accuracy of the model: {}".format(correct * 1. / total))

    def save_model_state(self, model_path):
        torch.save(self.model.state_dict(), model_path)
