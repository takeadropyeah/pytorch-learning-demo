import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,6,5,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6,16,5,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,120,5,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(),

        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(120*1*1,84),
            nn.ReLU(),
            nn.Linear(84,num_classes),
        )

    def forward(self,x):
        x=self.features(x)
        x=x.flatten(1)
        return self.classifier(x)

model=LeNet5()
total=0
for name,p in model.named_parameters():
    print(f"{name:30s}{tuple(p.shape)!s:20s}{p.numel():>8,d}")
    total+=p.numel()
print(f"{'合计':30s}{'':20s}{total:>8,d}")