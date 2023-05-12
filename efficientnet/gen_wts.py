import torch
import struct
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b3')

model.eval()
with open('efficientnet-b3.wts', 'w') as f:
    f.write(f'{len(model.state_dict().keys())}\n')
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write(f'{k} {len(vr)} ')
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
