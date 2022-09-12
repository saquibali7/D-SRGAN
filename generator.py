import torch
import torch.nn as nn
import torchvision

class CNNBlocks(nn.Module):
  def __init__(self, in_channels ):
    super(CNNBlocks, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        nn.BatchNorm2d(in_channels),
        nn.PReLU(),
        nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        nn.BatchNorm2d(in_channels),
    )

  def forward(self, x):
      return self.conv(x)+x


class PixelShuffle(nn.Module):
  def __init__(self, in_channels, out_channels, upscale_factor):
    super(PixelShuffle, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, 3, 1, 1),
        nn.PixelShuffle(upscale_factor),
        nn.PReLU(),
    )

  def forward(self,x):
    return self.conv(x)



class Generator(nn.Module):
  def __init__(self, in_channels, features):
    super(Generator, self).__init__()
    self.first_layer = nn.Sequential(
        nn.Conv2d(in_channels, features, 3, 1, 1),
        nn.PReLU(),
    )
    self.RB1 = CNNBlocks(features)
    self.RB2 = CNNBlocks(features)
    self.RB3 = CNNBlocks(features)
    self.RB4 = CNNBlocks(features)

    self.mid_layer = nn.Sequential(
        nn.Conv2d(features, features*4, 3, 1, 1),
        nn.PReLU(),
    )
    self.PS1 = PixelShuffle(features*4, features*8,2)
    self.PS2 = PixelShuffle(features*2, features*4,2)

    self.final_layer = nn.Sequential(
        nn.Conv2d(features, in_channels, 3, 1, 1),
        nn.Tanh(),
    )


  def forward(self, x):
    x1 = self.first_layer(x)
    x2 = self.RB1(x1)
    x3 = self.RB2(x2)
    x4 = self.RB3(x3)
    x5 = self.RB4(x4)
    x6 = self.mid_layer(x5+x1)
    x7 = self.PS1(x6)
    x8 = self.PS2(x7)
    return self.final_layer(x8)

def test():
  gen = Generator(1, 128)
  x = torch.randn(8,1,128,128)
  out = gen(x)
  print(out.shape)


if __name__ == "__main__":
    test()  