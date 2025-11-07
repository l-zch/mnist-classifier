import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchinfo import summary


loss_func = F.cross_entropy

def accuracy(input:torch.Tensor, target:torch.Tensor):
    preds = torch.argmax(input, dim=1)
    return (preds == target).float().mean()

def evaluate(model:nn.Module, dataloader: DataLoader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            total_acc += accuracy(preds, y_batch)
            total_loss += loss_func(preds, y_batch)
    return total_loss / len(dataloader), total_acc / len(dataloader)

def fit(model:nn.Module, optimizer:optim.Optimizer, dataloader:DataLoader, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        preds = model(x_batch)
        loss = loss_func(preds, y_batch)
        total_loss += loss
        total_acc += accuracy(preds, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader), total_acc / len(dataloader)

class TrainRunner:
    def __init__(self, train_data, log_path:Path):
        self.train_data = train_data
        self.log_path = log_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        

    def train(self, model, lr, bs, epochs, name=None):
        g = torch.Generator().manual_seed(42)
        train_data, val_data = random_split(self.train_data, [50000, 10000], generator=g)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=bs*2, shuffle=True)

        time_str = f'{datetime.now().timestamp():.0f}'
        name = name or time_str
        log_dir = self.log_path / name
        if log_dir.exists():
            log_dir = self.log_path / (name + time_str)
            
        writer = SummaryWriter(log_dir=log_dir)
        
        with open(log_dir/"summary.txt", "w") as f:
            f.write(f"leaning_rate={lr}, ")
            f.write(f"batch_size={bs}\n")
            f.write(str(summary(
                model,
                input_data=next(iter(train_loader))[0],
                col_width=20,
                row_settings=["var_names"]
            )))

        # images = next(iter(train_loader))[0]
        # model.eval()
        # example_input = images.to(next(model.parameters()).device)

        # with torch.no_grad():
        #     traced = torch.jit.trace(
        #         model,
        #         example_input,
        #         strict=False
        #     )

        # graph = traced.graph

        # writer._get_file_writer().add_graph(graph)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

        model.to(self.device)
        for epoch in range(1, epochs+1):
            train_loss, train_acc = fit(model, optimizer, train_loader, self.device)
            val_loss, val_acc = evaluate(model, val_loader, self.device)
            print(f"epoch {epoch} | loss: {train_loss:.4f}, acc: {train_acc:.4f} loss (val):{val_loss:.4f}, acc (val): {val_acc:.4f}")
            writer.add_scalars(
                "Loss", {"train": train_loss, "val": val_loss}, epoch
            )
            writer.add_scalars(
                "Acc", {"train": train_acc, "val": val_acc}, epoch
            )
            writer.flush()

        writer.close()
    
    def quick_validation(self, model):
        loader = DataLoader(self.train_data)
        print(summary(
            model,
            input_data=next(iter(loader))[0],
            col_width=20,
            row_settings=["var_names"]
        ))
    