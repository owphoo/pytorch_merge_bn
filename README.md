# pytorch_merge_bn
## pytorch merge bn

1. Run 
```
python pytorch_merge_bn.py YOUR_PYTORCH_MODEL
```

2. Run the merged bn model
```
# You should set params **bias=True** in nn.Conv2d or nn.ConvTranspose2d first
net = Your_Net()
net = net.cuda()
net.eval()
checkpoint = torch.load(MERGE_BN_MODEL)
net.load_state_dict(checkpoint['net_state_dict'])
...
```
