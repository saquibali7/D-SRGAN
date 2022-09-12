import torch
import torch.nn as nn
import torchvision


class Blocks(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super(Blocks, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )

  def forward(self, x):
    return self.conv(x)  

class Discriminator(nn.Module):
  def __init__(self, in_channels, features):
    super(Discriminator, self).__init__()
    self.first_layer= nn.Sequential(
        nn.Conv2d(in_channels, features, 3, 2 ,1),
        nn.LeakyReLU(0.2),
    )
    self.Block1 = Blocks(features, features*2, stride=2)
    self.Block2 = Blocks(features*2, features*2, stride=1)
    self.Block3 = Blocks(features*2, features*4, stride=2)
    self.Block4 = Blocks(features*4, features*4, stride=1)
    self.Block5 = Blocks(features*4, features*8, stride=2)
    self.Block6 = Blocks(features*8, features*8, stride=1)
    self.Block7 = Blocks(features*8, features*8, stride=2)
    self.Block8 = Blocks(features*8, features*8, stride=2)
    self.Block9 = nn.Sequential(
        nn.Conv2d(features*8, features*4, 3, 2, 1),
        
        nn.LeakyReLU(0.2),
    )
    self.final_layer = nn.Sequential(
        nn.Linear(features*4, 1),
        nn.Sigmoid(),
    )

  def forward(self, x):
    x =  self.first_layer(x)
    x =  self.Block1(x)
    x =  self.Block2(x)
    x =  self.Block3(x)
    x =  self.Block4(x)
    x =  self.Block5(x)
    x =  self.Block6(x)
    x =  self.Block7(x)
    x =  self.Block8(x)
    x = self.Block9(x)
    x = x.view(x.size(0), -1)
    return self.final_layer(x)

def test():
  x = torch.randn(8,1,128,128)
  disc = Discriminator(1,128)
  out = disc(x)
  print(out.shape)

if __name__ == "__main__":
    test()    