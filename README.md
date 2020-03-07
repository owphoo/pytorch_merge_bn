# pytorch_merge_bn
## pytorch merge bn

1. Run 
```
python pytorch_merge_bn.py YOUR_PYTORCH_MODEL
```

### 2. Run the merged bn model

2.1 You should set params **bias=True** in **nn.Conv2d** and **nn.ConvTranspose2d** and remove **nn.BatchNorm2d** first like: 
```
def conv_bn_relu(...):
    return  nn.Sequential(
            nn.Conv2d(bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
    )

```

2.2 Load the net 
```

net = YourNet()
net = net.cuda()
net.eval()
checkpoint = torch.load(MERGE_BN_MODEL)
net.load_state_dict(checkpoint['net_state_dict'])
...
```
