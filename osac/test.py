# import torch
# print(torch.cuda.is_available())  # Should print: True
# print(torch.cuda.get_device_name(0)) # Should show your GPU name

from rembg import remove
from PIL import Image

inpth = '20368979_786370338210664_539612317845850644_o.jpg'
otpth = 'cropped.png'

inp = Image.open(inpth)
output = remove(inp)
output.save(otpth)
Image.open('cropped.png')
