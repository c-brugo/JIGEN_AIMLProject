import argparse

import torch
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
from data.DatasetEdit import DatasetEdit
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch rotation training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--path_dataset", default="JIGEN_AIMLProject/", help="Path where the dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float,help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--n_perm", "-p", type=int, default=3, help="Number of rotations possible (excluded the ordered one)")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="jigen_resnet18", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight of jigsaw resolution")
    parser.add_argument("--alpha_t", type=float, default=0.7, help="Weight of jigsaw resolution for target domain data")
    parser.add_argument("--beta", type=float, default=0.4, help="Fraction of shuffled images in the dataset")
    parser.add_argument("--eta", type=float, default=0.1, help="Weight of DA target empirical entropy loss")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    return parser.parse_args()

def emp_entropy_loss(x):
    return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        model = model_factory.get_network(args.network)(classes=args.n_classes, permutations = args.n_perm)
        self.model = model.to(device)

        self.rotation_editor = DatasetEdit(img_dim = args.image_size, P = args.n_perm)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, beta = args.beta, self_sup_transformer = self.rotation_editor.randomRotation, DA = True)
        self.target_loader = data_helper.get_val_dataloader(args)


        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all)

        self.n_classes = args.n_classes
        self.n_perm = args.n_perm

        self.alpha = args.alpha
        self.alpha_t = args.alpha_t
        self.eta = args.eta

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, (data, class_l, perm_l) in enumerate(self.source_loader):

            data, class_l, perm_l = data.to(self.device), class_l.to(self.device), perm_l.to(self.device)

            self.optimizer.zero_grad()

            #calculated class loss only with perm_l == 0 and class_l!=-1 (ordered images not from target domain)
            mask = perm_l == 0
            data_ordered = data[torch.nonzero(mask)][:,0]
            class_l_ordered = class_l[torch.nonzero(mask)][:, 0]

            mask_ordered_source = class_l_ordered != -1
            data_ordered_source = data_ordered[torch.nonzero(mask_ordered_source)][:,0]
            class_l_ordered_source = class_l_ordered[torch.nonzero(mask_ordered_source)][:, 0]


            #CLASS LOSS
            class_logit = self.model(data_ordered_source)
            class_loss = criterion(class_logit, class_l_ordered_source)
            _, cls_pred = class_logit.max(dim=1)

            loss = class_loss
            #loss.backward()


            #CLASS EMPIRICAL ENTROPY LOSS TARGET
            mask_ordered_target = class_l_ordered == -1
            if torch.nonzero(mask_ordered_target).size()[0]>0:
                data_ordered_target = data_ordered[torch.nonzero(mask_ordered_target)][:,0]
                class_target_logit = self.model(data_ordered_target)
                emp_entropy_target_loss = emp_entropy_loss(class_target_logit)

                loss = loss + self.eta * emp_entropy_target_loss
                del class_target_logit
            else:
                emp_entropy_target_loss = torch.tensor(0)




            #JIGSAW LOSS (SOURCE)
            if self.alpha != 0:
                mask_source = class_l != -1
                data_source = data[torch.nonzero(mask_source)][:,0]
                perm_l_source = perm_l[torch.nonzero(mask_source)][:,0]

                perm_source_logit = self.model(data_source, alpha = self.alpha)
                perm_source_loss = criterion(perm_source_logit, perm_l_source)
                _, perm_source_pred = perm_source_logit.max(dim=1)

                loss = loss + self.alpha * perm_source_loss
                #loss.backward()


            #JIGSAW LOSS (TARGET)
            if self.alpha_t != 0:
                mask_target = class_l == -1
                data_target = data[torch.nonzero(mask_target)][:,0]
                perm_l_target = perm_l[torch.nonzero(mask_target)][:,0]

                perm_target_logit = self.model(data_target, alpha = self.alpha_t)
                perm_target_loss = criterion(perm_target_logit, perm_l_target)
                _, perm_target_pred = perm_target_logit.max(dim=1)

                loss = loss + self.alpha_t * perm_target_loss
                #loss.backward()


            loss.backward()
            self.optimizer.step()

            class_l = class_l_ordered_source
            losses_log = {"Class Loss ": class_loss.item(), "Empirical Entropy Target Loss": emp_entropy_target_loss.item(), "Total Loss ": loss.item()}
            accuracy_log = {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item()}

            if self.alpha != 0:
                losses_log["Rotation Source Loss "] = perm_source_loss.item()
                accuracy_log["Rotation Source Accuracy "] = torch.sum(perm_source_pred == perm_l_source.data).item()
            if self.alpha_t != 0:
                losses_log["Rotation Target Loss "] = perm_target_loss.item()
                accuracy_log["Rotation Target Accuracy "] = torch.sum(perm_target_pred == perm_l_target.data).item()

            self.logger.log(it, len(self.source_loader),
                            losses_log,
                            accuracy_log,
                            data.shape[0])
            del loss, class_loss, class_logit, emp_entropy_target_loss
            if self.alpha != 0:
                del perm_source_loss, perm_source_logit
            if self.alpha_t != 0:
                del perm_target_loss, perm_target_logit

        if False:
            self.model.eval()
            with torch.no_grad():
                for phase, loader in self.test_loaders.items():
                    total = len(loader.dataset)
                    class_correct = self.do_test(loader)
                    class_acc = float(class_correct) / total
                    self.logger.log_test(phase, {"Classification Accuracy": class_acc})
                    self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, (data, class_l) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct
    
    def final_test(self):
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                print("\"final_{}\" : {}".format(str(phase), class_acc))

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=100)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
            self.scheduler.step()

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("{\"best_val\" : %g, \"best_test\" : %g}" % (val_res.max(), test_res.max()))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()
    trainer.final_test()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
