'''
feature map size can varry,important thing to watch out for size matching post upscaling
torch.Size([1, 3, 384, 1280])
torch.Size([1, 16, 192, 640])
torch.Size([1, 24, 96, 320])
torch.Size([1, 48, 48, 160])
torch.Size([1, 88, 24, 80])
torch.Size([1, 120, 24, 80])
torch.Size([1, 208, 12, 40])
torch.Size([1, 352, 12, 40])
torch.Size([1, 1408, 12, 40])
prerequisite is install efficientNet pytorch from  https://github.com/lukemelas/EfficientNet-PyTorch
'''
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientNet_base(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNet_base, self).__init__(blocks_args=blocks_args, global_params=global_params)
    
    

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        P = []
        index = 0
        num_repeat = 0
        size1=0
        # Blocks
       
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            
             
            #print(tmp.size())
            if size1 != x.size(1):
                
                P.append(x)
                size1=x.size(1)
                
           
        return P
    
            
        
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        # Convolution layers
        P = self.extract_features(inputs)
        return P
 
 class Efficientnet_lat(nn.Module):
    '''Mixture of previous classes'''
    
    def set_dropout(self,model, drop_rate):
    # source: https://discuss.pytorch.org/t/how-to-increase-dropout-rate-during-training/58107/4
        for name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
                print("name:", name)
                print("children:\n", child)
        return model            
    
    def __init__(self, n_classes,dropout=0.2):
        super(Efficientnet_lat, self).__init__()
       
        self.mix = nn.Parameter(torch.FloatTensor(12)) #linear mix of features at all levels this can be turned off
        self.mix.data.fill_(1)
       
        self.dropout=dropout
        self.base_model = EfficientNet_base.from_pretrained('efficientnet-b2')
        self.base_model=self.set_dropout(self.base_model, self.dropout)
       
       # Lateral layers convert effnet outputs to a common feature size
        
        #16, 24, 48, 88, 120, 208, 352
        self.lat16 = nn.Conv2d(16, 256, 1)
        self.lat24 = nn.Conv2d(24, 256, 1)
        self.lat48 = nn.Conv2d(48, 256, 1)
        self.lat88 = nn.Conv2d(88, 256, 1)
        self.lat120 = nn.Conv2d(120, 256, 1)
        self.lat208 = nn.Conv2d(208, 256, 1)
        self.lat352 = nn.Conv2d(1408, 256, 1)
        
        self.bn24 = nn.GroupNorm(16, 256)
        self.bn16 = nn.GroupNorm(16, 256)
        self.bn48 = nn.GroupNorm(16, 256)
        self.bn88 = nn.GroupNorm(16, 256)
        self.bn120 = nn.GroupNorm(16, 256)
        self.bn208 = nn.GroupNorm(16, 256)
        self.bn352 = nn.GroupNorm(16, 256)
        self.mp = nn.MaxPool2d(2)
   def forward(self, x):
   
        features = self.base_model.extract_features(x)#get list of features
       
        lat16 = self.mp(F.relu(self.bn16(self.lat16(features[0])))) #64
        
      
        
        lat24 = self.mp(F.relu(self.bn24(self.lat24(features[1]))))
        lat24up=nn.Upsample(scale_factor=2)(lat24)
        lat_24_16cat= lat24up*self.mix[0]+lat16*self.mix[1] #16*12
        lat_24_16cat=self.mp(lat_24_16cat) # 8*6
        
        
        
        lat48 = self.mp(F.relu(self.bn48(self.lat48(features[2]))))
        lat48up=nn.Upsample(scale_factor=2)(lat48) #do umsample to match to size of feature map of previous layer
        lat_48_24cat= lat48up*self.mix[2]+lat_24_16cat*self.mix[3]
        lat_48_24cat=self.mp(lat_24_16cat) #4*3
        
        lat88 = self.mp(F.relu(self.bn88(self.lat88(features[3]))))
        lat88up=nn.Upsample(scale_factor=2)(lat88)
        lat_88_48cat= lat88up*self.mix[4]+lat_48_24cat*self.mix[5] #4*3
        lat_88_48cat=self.mp(lat_88_48cat) #12 40
        
        lat120 = self.mp(F.relu(self.bn120(self.lat120(features[4])))) #upsample not needed as 88 and 120 have got same feature map
       
        lat_120_88cat= lat120*self.mix[6]+lat_88_48cat*self.mix[7] #4*3
        
        
        
        lat208 = self.mp(F.relu(self.bn208(self.lat208(features[5])))) #2*1
        lat208up=nn.Upsample(scale_factor=2)(lat208)
        lat_208_88cat= lat208up*self.mix[8]+lat_120_88cat*self.mix[9] #2*1
       
        
        feats= self.base_model._swish(self.base_model._bn1(self.base_model._conv_head(features[6])))  #1408
        lat352 = (F.relu(self.bn352(self.lat352(feats)))) #2*1 eature maps is same for conv head and last block
        feats= lat352*self.mix[10]+lat_208_88cat*self.mix[11] #2*1
        feats=self.mp(feats)
        
        
        
        # Add positional info
       
        return feats
