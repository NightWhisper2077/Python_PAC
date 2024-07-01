import os
import PIL
import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, random_split
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

transform = transforms.Compose([
    transforms.RandomRotation([-90, 90]),
    transforms.ColorJitter(),
    transforms.RandomAdjustSharpness(0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = create_dataset(csv_file, image_dir, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]


class R2Loss(nn.Module):
    def forward(self, target, pred):
        ss_total = torch.sum((target - torch.mean(target, dim=0)) ** 2, dim=0)
        ss_residual = torch.sum((target - pred) ** 2, dim=0)
        r2_scores = ss_residual / ss_total
        mean_r2_score = torch.mean(r2_scores)
        return mean_r2_score


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
        self.lin1 = nn.Linear(1163, 512)
        self.rb1 = ResidualBlock(512, 512)
        self.rb2 = ResidualBlock(128, 128)
        self.lin2 = nn.Linear(512, 128)
        self.lin3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        nn.init.kaiming_normal_(self.lin1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.lin2.weight, nonlinearity='relu')
        init.zeros_(self.lin1.bias)
        init.zeros_(self.lin2.bias)

    def forward(self, h3):
        h3 = self.lin1(h3)
        h3 = self.bn1(h3)
        h3 = self.relu(h3)
        h3 = self.dropout1(h3)
        h3 = self.rb1(h3)
        h3 = self.dropout2(h3)
        h3 = self.lin2(h3)
        h3 = self.bn2(h3)
        h3 = self.relu(h3)
        h3 = self.rb2(h3)
        h3 = self.lin3(h3)

        return h3


class ClusterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modelka = efficientnet_v2_s(weights=True)

        self.heads = nn.ModuleList()
        for _ in range(6):
            head = nn.Sequential(
                OneModel()
            )
            self.heads.append(head)

    def forward(self, x1, x2):
        with torch.no_grad():
            h1 = self.modelka(x1)

        h2 = x2

        h3 = torch.cat((h1, h2), dim=1)

        outputs = [head(h3) for head in self.heads]

        return outputs

    def eval_model(self):
        self.modelka.eval()

    def train_model(self):
        self.modelka.train()


def create_dataset(csv_path, folder_path, preprocess):
    scaler = StandardScaler()
    df = pd.read_csv(csv_path)

    cols = df.columns

    x = pd.concat(
        [pd.DataFrame(scaler.fit_transform(df[cols[1:164]]), columns=cols[1:164].to_list()), df[cols[164:170]]], axis=1)
    x.index = df[cols[0]].to_list()

    res = []
    index = 0
    prc = 0
    step = 1

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = preprocess(PIL.Image.open(img_path))
        img_id = int(filename[0:len(filename) - 5])

        if img_id in x.index:
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


def train_step(train_loader, model, criterion, optimizer):
    model.train()
    model.eval_model()

    running_loss = [0, 0, 0, 0, 0, 0]
    running_score = [0, 0, 0, 0, 0, 0]

    ind = 0

    size = len(train_loader)

    for images, features, targets in train_loader:
        images = images.cuda()
        features = features.cuda()
        targets = targets.unsqueeze(-1).transpose(0, 1)
        targets = [targets[0].cuda(), targets[1].cuda(), targets[2].cuda(),
                   targets[3].cuda(), targets[4].cuda(), targets[5].cuda()]

        optimizer.zero_grad()

        outputs = model(images, features)

        losses = []

        for i in range(6):
            loss = criterion(targets[i], outputs[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)

        result = []
        for x, y in zip(running_loss, losses):
            result.append(x + y.item())

        running_loss = result

        outputs = [outputs[0].cpu().detach().numpy(), outputs[1].cpu().detach().numpy(),
                   outputs[2].cpu().detach().numpy(),
                   outputs[3].cpu().detach().numpy(), outputs[4].cpu().detach().numpy(),
                   outputs[5].cpu().detach().numpy()]

        targets = [targets[0].cpu().detach().numpy(), targets[1].cpu().detach().numpy(),
                   targets[2].cpu().detach().numpy(),
                   targets[3].cpu().detach().numpy(), targets[4].cpu().detach().numpy(),
                   targets[5].cpu().detach().numpy()]

        score = [r2_score(targets[0], outputs[0]), r2_score(targets[1], outputs[1]), r2_score(targets[2], outputs[2]),
                 r2_score(targets[3], outputs[3]), r2_score(targets[4], outputs[4]), r2_score(targets[5], outputs[5])]

        result1 = []
        for x, y in zip(running_score, score):
            result1.append(x + y.item())

        running_score = result1

        ind += 1

        prc = int((ind / size) * 100)

        print(f'\r|', end='')

        for _ in range(prc):
            print(f'=', end='')

        for _ in range(100 - prc):
            print(' ', end='')

        print(f'|   Losses: {score[0]:.4f} {losses[0]:.4f} {prc}% ', end='')

    print('')

    with torch.no_grad():
        train_loss = sum(running_loss) / (6 * len(train_loader))
        train_score = sum(running_score) / (6 * len(train_loader))
    return train_loss, train_score, running_loss, running_score


def valid_step(val_loader, model, criterion):
    model.eval()

    running_loss = [0, 0, 0, 0, 0, 0]
    running_score = [0, 0, 0, 0, 0, 0]

    for images, features, targets in val_loader:
        images = images.cuda()
        features = features.cuda()
        targets = targets.unsqueeze(-1).transpose(0, 1)
        targets = [targets[0].cuda(), targets[1].cuda(), targets[2].cuda(),
                   targets[3].cuda(), targets[4].cuda(), targets[5].cuda()]

        outputs = model(images, features)

        losses = []

        for i in range(6):
            loss = criterion(targets[i], outputs[i])
            losses.append(loss)

        result = []
        for x, y in zip(running_loss, losses):
            result.append(x + y.item())

        running_loss = result

        outputs = [outputs[0].cpu().detach().numpy(), outputs[1].cpu().detach().numpy(),
                   outputs[2].cpu().detach().numpy(),
                   outputs[3].cpu().detach().numpy(), outputs[4].cpu().detach().numpy(),
                   outputs[5].cpu().detach().numpy()]

        targets = [targets[0].cpu().detach().numpy(), targets[1].cpu().detach().numpy(),
                   targets[2].cpu().detach().numpy(),
                   targets[3].cpu().detach().numpy(), targets[4].cpu().detach().numpy(),
                   targets[5].cpu().detach().numpy()]

        score = [r2_score(targets[0], outputs[0]), r2_score(targets[1], outputs[1]), r2_score(targets[2], outputs[2]),
                 r2_score(targets[3], outputs[3]), r2_score(targets[4], outputs[4]), r2_score(targets[5], outputs[5])]

        result1 = []
        for x, y in zip(running_score, score):
            result1.append(x + y.item())

        running_score = result1

    with torch.no_grad():
        val_loss = sum(running_loss) / (6 * len(val_loader))
        val_score = sum(running_score) / (6 * len(val_loader))
    return val_loss, val_score, running_loss, running_score


def train(train_loader, val_loader, epochs, model, criterion, optimizer, writer, writer2):
    best_score = -100
    train_losses = []
    valid_losses = []

    for i in range(epochs):
        train_loss, train_score, train_running_loss, train_running_score = train_step(train_loader, model, criterion,
                                                                                      optimizer)
        valid_loss, valid_score, val_running_loss, val_running_score = valid_step(val_loader, model, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Current epoch is {i}. Avg. train loss: {train_loss:.4f}, score: {train_score:.4f}')

        train_running_loss = [x / len(train_loader) for x in train_running_loss]
        train_running_score = [x / len(train_loader) for x in train_running_score]
        print(f'For features: {train_running_loss} | {train_running_score}')
        print(f'Current epoch is {i}. Avg. valid loss: {valid_loss:.4f}, score: {valid_score:.4f}')
        val_running_loss = [x / len(val_loader) for x in val_running_loss]
        val_running_score = [x / len(val_loader) for x in val_running_score]
        print(f'For features: {val_running_loss} | {val_running_score}')

        writer.add_scalar('training score', train_score, i)
        writer2.add_scalar('valid score', valid_score, i)
        
        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), 'best_score0001.pth')
            
            print("Best Score Model Saved!")

    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(args.loss_path)
    plt.show()
    return train_losses


dataset = CustomDataset(csv_file='newtable.csv', image_dir='train_images', transform=transform)

train_size = args.size - args.valid_size
val_size = args.valid_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

model = ClusterModel()
model.cuda()

criterion = R2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)

writer = SummaryWriter('runs/train/new')
writer2 = SummaryWriter('runs/train/new')

train(train_loader, val_loader, args.epoch, model, criterion, optimizer, writer, writer2)

writer.close()
writer2.close()

torch.save(model.state_dict(), 'my_model3.pth')
