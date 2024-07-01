import os
import PIL
import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import efficientnet_v2_s
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    prog='KaggleModel',
    description='PredictPlantsParams',
    epilog='I am bebra')

parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-v', '--valid_size', type=int)
parser.add_argument('-e', '--epoch', type=int)
parser.add_argument('-s', '--size', type=int)
parser.add_argument('-l', '--lr', type=float)
parser.add_argument('-t', '--loss_path')
# parser.add_argument('-p', '--score_path')
args = parser.parse_args()


class R2Loss(nn.Module):
    def forward(self, target, pred):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        r2 = ss_res / ss_tot
        return r2


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class OneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modelka = efficientnet_v2_s(weights=True)
        self.lin1 = nn.Linear(1163, 512)
        self.rb1 = ResidualBlock(512, 512)
        self.lin2 = nn.Linear(512, 128)
        self.lin3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_normal_(self.lin1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.lin2.weight, nonlinearity='relu')
        init.zeros_(self.lin1.bias)
        init.zeros_(self.lin2.bias)

    def forward(self, x1, x2):
        with torch.no_grad():
            h1 = self.modelka(x1)

        h2 = x2

        h3 = torch.cat((h1, h2), dim=1)

        h3 = self.lin1(h3)
        h3 = self.bn1(h3)
        h3 = self.relu(h3)
        h3 = self.rb1(h3)
        h3 = self.lin2(h3)
        h3 = self.bn2(h3)
        h3 = self.relu(h3)
        h3 = self.dropout(h3)
        h3 = self.lin3(h3)

        return h3

    def eval_model(self):
        self.modelka.eval()

    def train_model(self):
        self.modelka.train()


def create_dataset(csv_path, folder_path):
    scaler = StandardScaler()
    df = pd.read_csv(csv_path)

    cols = df.columns
    img_id = df[cols[0]].to_list()

    x = pd.concat(
        [pd.DataFrame(scaler.fit_transform(df[cols[1:164]]), columns=cols[1:164].to_list()), df[cols[164:170]]], axis=1)
    x.index = df[cols[0]].to_list()

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    res = []
    index = 0
    prc = 0
    step = 1

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = preprocess(PIL.Image.open(img_path))
        img_id = int(filename[0:len(filename) - 5])
        res.append([img, torch.tensor((x.loc[img_id])[cols[1:164]].tolist()),
                    torch.tensor((x.loc[img_id])[cols[164:170]].tolist())])

        index += 1

        if index == args.size:
            break

        if prc <= int((index / args.size) * 100) <= prc + step:
            print(f'\rDataset analysis: {int((index / args.size) * 100)}%   |', end='')

            for _ in range(prc):
                print(f'=', end='')

            for _ in range(100 - prc):
                print(' ', end='')

            print(f'|', end='')

            prc += step

    print('READY')

    return res


def create_batch(x, btch_size):
    data_size = len(x)

    cnt_batch = data_size // btch_size

    x_batch = []

    for i in range(cnt_batch):
        x_batch.append(x[i * btch_size: (i + 1) * btch_size])

    if data_size % btch_size != 0:
        x_batch.append(x[cnt_batch * btch_size: data_size])

    return x_batch


def train_step(x, model, criterion, optimizer):
    model.train()
    model.eval_model()

    running_loss = 0
    running_score = 0

    ind = 0

    size = len(x)

    for batch in x:
        x_train_img = []
        x_train_ft = []
        y_train = []

        for b in batch:
            x_train_img.append(b[0])
            x_train_ft.append(b[1])
            y_train.append(b[2])

        x_train_img = torch.stack(x_train_img).cuda()
        x_train_ft = torch.stack(x_train_ft).cuda()
        y_train = torch.stack(y_train).transpose(0, 1)[0].unsqueeze(1).cuda()

        optimizer.zero_grad()

        output = model(x_train_img, x_train_ft)

        loss = criterion(y_train, output)
        loss.backward()
        optimizer.step()
        running_loss += loss

        output_cpu = output.cpu().detach().numpy()
        y_cpu = y_train.cpu().detach().numpy()

        del x_train_img
        del x_train_ft
        del y_train

        torch.cuda.empty_cache()

        score = r2_score(y_cpu, output_cpu)

        running_score += score

        ind += 1

        prc = int((ind / size) * 100)

        print(f'\r|', end='')

        for _ in range(prc):
            print(f'=', end='')

        for _ in range(100 - prc):
            print(' ', end='')

        print(f'|   Losses: {score:.4f} {loss:.4f} {prc}% ', end='')

    print('')

    with torch.no_grad():
        train_loss = running_loss / len(x)
        train_score = running_score / len(x)
    return train_loss.item(), train_score.item()


def valid_step(x, model, criterion):
    model.eval()

    running_loss = 0
    running_score = 0

    for batch in x:
        x_valid_img = []
        x_valid_ft = []
        y_valid = []

        for b in batch:
            x_valid_img.append(b[0])
            x_valid_ft.append(b[1])
            y_valid.append(b[2])

        x_valid_img = torch.stack(x_valid_img).cuda()
        x_valid_ft = torch.stack(x_valid_ft).cuda()
        y_valid = torch.stack(y_valid).transpose(0, 1)[0].unsqueeze(1).cuda()

        output = model(x_valid_img, x_valid_ft)

        loss = criterion(y_valid, output)
        running_loss += loss

        output_cpu = output.cpu().detach().numpy()
        y_cpu = y_valid.cpu().detach().numpy()

        score = r2_score(y_cpu, output_cpu)

        running_score += score

    with torch.no_grad():
        valid_loss = running_loss / len(x)
        valid_score = running_score / len(x)
    return valid_loss.item(), valid_score.item()


def train(x, x_val, epochs, model, criterion, optimizer, writer, writer2):
    train_losses = []
    valid_losses = []
    train_score = []
    valid_score = []

    for i in range(epochs):
        train_loss, train_score = train_step(x, model, criterion, optimizer)
        valid_loss, valid_score = valid_step(x_val, model, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Current epoch is {i}. Avg. train loss: {train_loss:.4f}, score: {train_score:.4f}')
        print(f'Current epoch is {i}. Avg. valid loss: {valid_loss:.4f}, score: {valid_score:.4f}')

        writer.add_scalar('training score', train_score, i)
        writer2.add_scalar('valid score', valid_score, i)

    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(args.loss_path)
    plt.show()
    return train_losses


x_train = create_dataset('train.csv', 'train_images')
x_val = x_train[0:args.valid_size]
x_train = x_train[args.valid_size:args.size]

x_train_batch = create_batch(x_train, args.batch_size)
x_val_batch = create_batch(x_val, args.batch_size)

model = OneModel()
model.cuda()

criterion = R2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

writer = SummaryWriter('runs/train/new')
writer2 = SummaryWriter('runs/train/new')

train(x_train_batch, x_val_batch, args.epoch, model, criterion, optimizer, writer, writer2)

writer.close()
writer2.close()

torch.save(model.state_dict(), 'my_model2.pth')
