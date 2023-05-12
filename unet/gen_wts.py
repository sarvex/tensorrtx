import torch
import sys
import struct

def main():
  device = torch.device('cpu')
  state_dict = torch.load(sys.argv[1], map_location=device)

  with open("unet.wts", 'w') as f:
    f.write(f"{len(state_dict.keys())}\n")
    for k, v in state_dict.items():
      print('key: ', k)
      print('value: ', v.shape)
      vr = v.reshape(-1).cpu().numpy()
      f.write(f"{k} {len(vr)}")
      for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
      f.write("\n")

if __name__ == '__main__':
  main()

