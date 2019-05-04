import torch
import os
from gan_model import *
from movie_data_loader import *
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

 
if __name__ == '__main__':
	Large = False
	ngf = 64
	PATH = 'G_epoch13.pth.tar'

	save_path = 'output'
	ori_path = 'demoOriginal'

	model_G = ConvGen(ngf)
	checkpoint_G = torch.load(PATH)
	model_G.load_state_dict(checkpoint_G['state_dict'])
	model_G.eval()

	val_loader = get_loader(os.path.join(ori_path, 'data/'),
		batch_size=16,
		large=Large,
		mode='val',
		num_workers=4,
		)
	val_bs = val_loader.batch_size
	cnt = 1
	with torch.no_grad(): # Fuck torch.no_grad!! Gradient will accumalte if you don't set torch.no_grad()!!
		for i, (data, target) in enumerate(val_loader):
			data, target = Variable(data), Variable(target)
			print(data.shape)
			fake =  model_G(data).data
			print("fake shape: ", fake.shape)
			for i in range(val_bs):
				# validate with fake
				pred = fake[i].cpu().numpy()
				# print (type(pred))

				pred_rgb = (np.transpose(pred, (1,2,0)).astype(np.float64) + 1) / 2.


				# print(type(pred_rgb))
				# print("pred shape: ", pred_rgb.shape)
				# im = Image.fromarray(pred_rgb, 'RGB')

				im = Image.fromarray(np.uint8(im), 'RGB')

				im = im.resize((480,360))

				im.save('output/'+ '%05d.png' %(cnt))

				# matplotlib.image.imsave('output/'+ '%05d.png' %(cnt), pred_rgb)


				# new_im = Image.fromarray(pred_rgb)
				# new_im.save("numpy_altered_sample2.png" % )
				# pred_rgb = pred_rgb.resize((480,360))
				# plt.figure(figsize=(480,360))
				# plt.imshow(pred_rgb)
				# plt.axis('off')
				# plt.tight_layout()
				# plt.savefig(save_path+ '%05d.png' %(cnt))
				# print(os.path.join(save_path, '%05d.png'))
				# pred_rgb.save("output/", '%05d.png' %(cnt))
				# im = Image.fromarray(pred_rgb)
				# im.save("your_file.jpeg")
				# im.save('output/'+ '%05d.png' %(cnt))
				cnt+=1

