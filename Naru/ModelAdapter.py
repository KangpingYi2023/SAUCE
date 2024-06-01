import torch
import torch.nn as nn
import math
import torch.optim as optim
import copy
from torch.nn import functional as F, init


class ModelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias


class ModelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        # self.heads = nn.ModuleList([LabelAdaptHead() for _ in range(num_head)])
        self.heads = ModelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = self.cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        # return sum([gate[:, i] * self.heads[i](y, inverse=inverse) for i in range(self.num_head)])
        return (gate * self.heads(y, inverse=inverse)).sum(-1)

    def cosine(x1, x2, eps=1e-8):
        x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
        x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
        return x1 @ x2.transpose(0, 1)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr_outer)

    def inner_update(self, task_data, model):
        # model_copy = copy.deepcopy(model)
        inner_optimizer = optim.SGD(self.model.parameters(), lr=self.lr_inner)

        for x, y in task_data:
            y_pred = self.model(x)
            loss = nn.MSELoss()(y_pred, y)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return self.model.state_dict()

    def meta_update(self, tasks):
        # meta_grads = []

        for task_data in tasks:
            original_state = {
                name: param.clone() for name, param in self.model.named_parameters()
            }

            inner_params = self.inner_update(task_data, model)

            outer_loss = 0
            for x, y in task_data:
                y_pred = self.model(x)
                outer_loss += nn.MSELoss()(y_pred, y)

            self.optimizer.zero_grad()
            outer_loss.backward()

            meta_grad = {
                name: param.grad for name, param in self.model.named_parameters()
            }
            for name, param in self.model.named_parameters():
                param.data -= self.lr_outer * meta_grad[name]


if __name__ == "__main__":
    model = SimpleModel()
    maml = MAML(model)

    tasks = [
        [(torch.tensor([1.0]), torch.tensor([2.0]))],
        [(torch.tensor([2.0]), torch.tensor([4.0]))],
        [(torch.tensor([4.0]), torch.tensor([8.0]))],
    ]

    for epoch in range(1000):  
        maml.meta_update(tasks)

    new_data = torch.tensor([3.0])
    prediction = model(new_data)
    print("prediction results:", prediction.item())
