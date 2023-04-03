import numpy as np
import glob, tqdm
from PIL import Image

from skimage import color
from skimage.filters import roberts, sobel_h, sobel_v

def grad(img):
	gx = sobel_h(img)
	gy = sobel_v(img)
	return np.dstack([gx, gy])

check = True
if not check:
	import torch
	import lpips
	loss_fn_alx = lpips.LPIPS(net='alex').cuda() # best forward scores
	loss_fn_sqz = lpips.LPIPS(net='squeeze').cuda()
	loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization

	import tensorflow as tf
	tf.compat.v1.disable_v2_behavior()

	im1 = tf.compat.v1.placeholder(tf.uint8, [None, None, None, None])
	im2 = tf.compat.v1.placeholder(tf.uint8, [None, None, None, None])
	im3 = tf.compat.v1.placeholder(tf.float32, [None, None, None, None])
	im4 = tf.compat.v1.placeholder(tf.float32, [None, None, None, None])
	psnr = tf.compat.v1.image.psnr(im1, im2, max_val=255)
	ssim_rgb = tf.compat.v1.image.ssim(im1, im2, max_val=255)
	ssim_yuv = tf.compat.v1.image.ssim(im3, im4, max_val=1.0)
	ssim_multi = tf.compat.v1.image.ssim_multiscale(im3, im4, max_val=1.0)
	sharp = tf.compat.v1.image.psnr(im3, im4, max_val=1.0)

	sess = tf.compat.v1.Session()

vids = ['000','003','004','005','015','016','018','025','032','033','035','074','077','078','081','082','083','084','086','088','103','110','114','117','158','162','163','165','176','178','199','200','202','203','212','214','217','223','228','240','257','261','267','279','280','284','285','287','289','290']
vids = [int(vid) for vid in vids]

# ours
pds = '../to_sel/2000/input_%03d_encoded_up_%02d.png'
gts = '../results_test300_up91perceptual_inv_2000/test50inv_all_frames_15frames/images/input_%03d_encoded_up_%02d.png'

# vid2vid
# pds = '/home/lzq/lzy/imaginaire/projects/vid2vid/output/london_xiaohu_testinv/stuttgart_00_%03d/stuttgart_00_000000_%06d_leftImg8bit.jpg'
# gts = '/home/lzq/lzy/imaginaire/projects/vid2vid/output/london_xiaohu_test_300_epoch5/stuttgart_00_%03d/stuttgart_00_000000_%06d_leftImg8bit.jpg'

# wc_vid2vid
# pds = '/home/lzq/lzy/imaginaire/projects/wc_vid2vid/output/london_xiaohu_testinv/stuttgart_00_%03d/fake/stuttgart_00_000000_%06d_leftImg8bit.jpg'
# gts = '/home/lzq/lzy/imaginaire/projects/wc_vid2vid/output/london_xiaohu_test_300_epoch27/stuttgart_00_%03d/fake/stuttgart_00_000000_%06d_leftImg8bit.jpg'

li_rgb, li_alx, li_sqz, li_vgg = [], [], [], []
li_psnr, li_ssim_rgb, li_ssim_yuv, li_ssim_multi, li_sharp = [], [], [], [], []
for seq, vid in tqdm.tqdm(list(enumerate(vids))):
	res = []
	for j in range(15):

		# wc_vid2vid / vid2vid
		# pd = np.array(Image.open(pds % (vid, j+2)).convert('RGB'))[:,-1024:]
		# pd = np.array( Image.fromarray(pd).resize((512,256),Image.LANCZOS) )
		# gt = np.array(Image.open(pds % (vid, 46-j)).convert('RGB'))[:,-1024:]
		# gt = np.array( Image.fromarray(gt).resize((512,256),Image.LANCZOS) )

		# gt = np.array(Image.open(gts % (vid, 16-j)).convert('RGB'))[:,-1024:]
		# gt = np.array( Image.fromarray(gt).resize((512,256),Image.LANCZOS) )
		# gt = np.hstack([gt[:,256:], gt[:,:256]])

		# ours
		pd = np.array(Image.open(pds % (vid, j)))
		gt = np.array(Image.open(gts % (seq, j)))

		if check:
			diff = np.abs(pd * 1.0 - gt * 1.0)
			res.append(Image.fromarray(np.vstack([pd, gt, diff.astype(np.uint8)])))
			print(vid, diff.mean())
			continue

		delta = ((pd.astype(np.float) - gt.astype(np.float)) ** 2).mean()

		pd_gray = color.rgb2yuv(pd)[..., 0] # 0~1
		gt_gray = color.rgb2yuv(gt)[..., 0] # 0~1

		pd_grad = (grad(pd_gray) + 1) / 2 # (-1~1) to (0~1)
		gt_grad = (grad(gt_gray) + 1) / 2 # (-1~1) to (0~1)

		psnr_val, rgb_val = sess.run([psnr, ssim_rgb], feed_dict={im1: gt[np.newaxis], im2: pd[np.newaxis]})
		yuv_val = sess.run(ssim_yuv, feed_dict={im3: gt_gray[np.newaxis,...,np.newaxis], im4: pd_gray[np.newaxis,...,np.newaxis]})
		multi_val = sess.run(ssim_multi, feed_dict={im3: gt_gray[np.newaxis,...,np.newaxis], im4: pd_gray[np.newaxis,...,np.newaxis]})
		sharp_val = sess.run(sharp, feed_dict={im3: gt_grad[np.newaxis], im4: pd_grad[np.newaxis]})

		li_rgb.append(delta)
		li_psnr.append(psnr_val)
		li_ssim_rgb.append(rgb_val)
		li_ssim_yuv.append(yuv_val)
		li_ssim_multi.append(multi_val)
		li_sharp.append(sharp_val)

		# pd = np.array(Image.open(pds % (i, j)).convert('RGB'))
		# gt = np.array(Image.open(gts[15*i+j]))[:,512:,:3]

		pd = torch.from_numpy(pd).permute([2,0,1]).float().cuda().unsqueeze(0) / 127.5 - 1
		gt = torch.from_numpy(gt).permute([2,0,1]).float().cuda().unsqueeze(0) / 127.5 - 1

		with torch.no_grad():
			li_alx.append(loss_fn_alx(gt, pd).detach().cpu().numpy()[0,0,0,0])
			li_sqz.append(loss_fn_sqz(gt, pd).detach().cpu().numpy()[0,0,0,0])
			li_vgg.append(loss_fn_vgg(gt, pd).detach().cpu().numpy()[0,0,0,0])

		print(
			vid,
			'%.4lf' % np.mean(li_rgb),
			'%.4lf' % np.mean(li_psnr),
			'%.4lf' % np.mean(li_ssim_rgb),
			'%.4lf' % np.mean(li_ssim_yuv),
			'%.4lf' % np.mean(li_ssim_multi),
			'%.4lf' % np.mean(li_sharp)
		)

		print(
			vid,
			'%.4lf' % np.mean(li_alx),
			'%.4lf' % np.mean(li_sqz),
			'%.4lf' % np.mean(li_vgg)
		)
	
	if check:
		res[0].save(f'{vid:03}.gif', append_images=res[1:], save_all=True, duaration=150, loop=0)
		# res[0].save(f'{i}.png')
		# input('press')

