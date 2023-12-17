import torch
import os
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from statistics import mean
from tqdm import tqdm

from model import EmotionalSpeechPredictor
from data_utils import AudioLabelLoader
from utils import (
    Hparam,
    save_checkpoint,
    load_checkpoint,
)

def main():
    hparams = Hparam("./config.yaml")
    train(hparams=hparams)

def train(hparams):
    global_step = 0
    global_eval_acc = 0
    global_eval_loss = 0

    if not hparams.train.use_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = AudioLabelLoader(hparams=hparams, set_name="train")
    train_loader = DataLoader(train_dataset, hparams.train.train_batch_size,
                                num_workers=8, shuffle=False, pin_memory=True)

    valid_dataset = AudioLabelLoader(hparams=hparams, set_name="valid")
    valid_loader = DataLoader(valid_dataset, hparams.train.eval_batch_size,
                                num_workers=8, shuffle=False, pin_memory=True)

    model = EmotionalSpeechPredictor(hparams=hparams).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.train.learning_rate,
                            betas=(0.9, 0.99), weight_decay=5e-4)

    start_epoch = 0
    if os.path.isfile(hparams.checkpoint.continue_once):
        model, optimizer, start_epoch = load_checkpoint(hparams.checkpoint.continue_once, model, optimizer)

    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(start_epoch, hparams.train.num_epochs):
        losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                
                if global_step % hparams.checkpoint.valid_interval == 0:
                    accuracy, valid_loss = evaluate(model, valid_loader, criterion, device=device)
                    if accuracy >= global_eval_acc:
                        global_eval_acc = accuracy
                        global_eval_loss = valid_loss
                        
                        if global_step != 0:
                            if not os.path.isdir(hparams.checkpoint.save_folder):
                                os.mkdir(hparams.checkpoint.save_folder)

                            checkpoint_path = os.path.join(hparams.checkpoint.save_folder, 
                                                            "model_{}_{}.pt".format(epoch, round(global_eval_acc, 2)))
                            save_checkpoint(model, optimizer, epoch, checkpoint_path)

                global_step += 1
                tepoch.set_postfix(train_loss=mean(losses), valid_loss=global_eval_loss, valid_acc=global_eval_acc)       


def evaluate(model, eval_loader, criterion, device='cpu'):
    model.eval()
    with torch.no_grad():
        y_ground_truth = []
        y_predict = []
        losses = []
        for x, y in eval_loader:
            x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            outputs = model(x)
            loss = criterion(outputs, y)
            losses.append(loss.item())

            idxs_pred = torch.argmax(outputs, dim=1)
            y_predict = y_predict + list(idxs_pred.cpu().detach().numpy())
            idxs_gt = torch.argmax(y, dim=1)
            y_ground_truth = y_ground_truth + list(idxs_gt.cpu().detach().numpy())

        correct = (np.array(y_ground_truth) == np.array(y_predict))
        accuracy = correct.sum() / correct.size
        loss = mean(losses)
        model.train()

        return accuracy, loss


if __name__ == "__main__":
    main()