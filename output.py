import torch
import os
from gan_model import *

 
if __name__ == '__main__':
	ngf = 64
	PATH = 'G_epoch13.pth.tar'

	save_path = 'output'
	ori_path = 'demoOriginal'

	model_G = ConvGen(ngf)
	checkpoint_G = torch.load(PATH)
	model_G.load_state_dict(checkpoint_G['state_dict'])
	model_G.eval()

	val_loader = get_loader(os.path.join(ori_path, 'data/'),
		batch_size=args.batch_size,
		large=args.large,
		mode='val',
		num_workers=4,
		)
	val_bs = val_loader.batch_size
	with torch.no_grad(): # Fuck torch.no_grad!! Gradient will accumalte if you don't set torch.no_grad()!!
		for i, (data, target) in enumerate(val_loader):
			data, target = Variable(data.cuda()), Variable(target.cuda())
			fake =  model_G(data).data
			for i in range(val_bs):
				# validate with fake
				pred = fake[i].cpu().numpy()
				pred_rgb = (np.transpose(pred, (1,2,0)).astype(np.float64) + 1) / 2.
				plt.figure(figsize=(480,360))
				plt.imshow(pred_rgb)
				plt.axis('off')
				plt.tight_layout()
				plt.savefig(os.path.join(data_root, '%05d.png'))