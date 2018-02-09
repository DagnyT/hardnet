This version of HardNet is trained on [PS dataset](https://github.com/rmitra/PS-Dataset) by Mitra et.al. with torch package and then converted to PyTorch. 

The structure of the network has minor changes, use this definition:
    
    class HardNetPS(nn.Module):
       def __init__(self):
            super(HardNetPS, self).__init__()
            self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias = True)
        )
        return
        def input_norm(self,x):
            flat = x.view(x.size(0), -1)
            mp = torch.mean(flat, dim=1)
            sp = torch.std(flat, dim=1) + 1e-7
            return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


If you use this weights, please cite:
    
    
    @ARTICLE{2018arXiv180101466M,
       author = {{Mitra}, R. and {Doiphode}, N. and {Gautam}, U. and {Narayan}, S. and 
        {Ahmed}, S. and {Chandran}, S. and {Jain}, A.},
        title = "{A Large Dataset for Improving Patch Matching}",
      journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1801.01466},
     primaryClass = "cs.CV",
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2018,
        month = jan,
       adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180101466M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
