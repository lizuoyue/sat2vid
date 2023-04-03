import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import numpy as np
import glob, tqdm
from PIL import Image

from skimage import color
from skimage.filters import roberts, sobel_h, sobel_v

import torch
import lpips
loss_fn_alx = lpips.LPIPS(net='alex').cuda() # best forward scores
loss_fn_sqz = lpips.LPIPS(net='squeeze').cuda()
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization

def grad(img):
	gx = sobel_h(img)
	gy = sobel_v(img)
	return np.dstack([gx, gy])

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

# pds = '../results_test_template_up11latest/test_template_all_frames_15frames/images/input_%03d_encoded_up_%02d.png'
# gts = sorted(glob.glob('../london_test_template/*.png'))

# vid2vid
# pds = '/home/lzq/lzy/imaginaire/projects/vid2vid/output/london_xiaohu_test_300_epoch5/stuttgart_00_%03d/stuttgart_00_000000_%06d_leftImg8bit.jpg'
pds = '/home/lzq/lzy/imaginaire/projects/vid2vid/output/london_xiaohu_testlateral/stuttgart_00_%03d/stuttgart_00_000000_%06d_leftImg8bit.jpg'
gts = sorted(glob.glob('../london_test300_15_frames/*.png'))

# wc_vid2vid
# pds = '/home/lzq/lzy/imaginaire/projects/wc_vid2vid/output/london_xiaohu_test_template_epoch27/stuttgart_00_%03d/fake/stuttgart_00_000000_%06d_leftImg8bit.jpg'
# pds = '/home/lzq/lzy/imaginaire/projects/wc_vid2vid/output/london_xiaohu_testlateral_epoch27/stuttgart_00_%03d/fake/stuttgart_00_000000_%06d_leftImg8bit.jpg'
# gts = sorted(glob.glob('../london_test300_15_frames/*.png'))

# our_cvpr
# pds = '../results_satemidas_pc_scnrn_coord/test300_all_frames_15frames/images/input_%03d_encoded_%02d.png'
# gts = sorted(glob.glob('../london_test300_15_frames/*.png'))

# ours_up11_latest
# pds = '../results_test300_up11latest/test300_all_frames_15frames/images/input_%03d_encoded_up_%02d.png'
# gts = sorted(glob.glob('../london_test300_15_frames/*.png'))

# ours_up17_perceptual
# pds = '../results_test300_up91perceptual_new300_lateral1000/test300_all_frames_15frames/images/input_%03d_encoded_up_%02d.png'
# gts = sorted(glob.glob('../london_test300_15_frames/*.png'))

# ours_template_up11
# pds = '../results_test_template_up11latest/test_template_all_frames_15frames/images/input_%03d_encoded_up_%02d.png'
# gts = sorted(glob.glob('../london_test_template/*.png'))

li_alx, li_sqz, li_vgg = [], [], []
li_psnr, li_ssim_rgb, li_ssim_yuv, li_ssim_multi, li_sharp = [], [], [], [], []
for i in tqdm.tqdm(list(range(0, 300))):
	res = []
	for j in range(15):#[7]:#

		# wc_vid2vid
		# pd = np.array(Image.open(pds % (i, j+2)).convert('RGB').resize((512,256),Image.LANCZOS))
		# gt = np.array(Image.open(gts[15*i+j]))[:,512:,:3]
		# gt = np.hstack([gt[:,128:], gt[:,:128]])

		# vid2vid
		pd = np.array(Image.open(pds % (i, j+2)).convert('RGB'))[:,-1024:]
		pd = np.array( Image.fromarray(pd).resize((512,256),Image.LANCZOS) )
		gt = np.array(Image.open(gts[15*i+j]))[:,512:,:3]
		gt = np.hstack([gt[:,128:], gt[:,:128]])

		# our_cvpr
		# pd = np.array(Image.open(pds % (i, j)).resize((512,256),Image.BICUBIC))
		# gt = np.array(Image.open(gts[15*i+j]))[:,512:,:3]

		# ours_up11_latest
		# pd = np.array(Image.open(pds % (i, j)).convert('RGB'))#.resize((256,128),Image.LANCZOS))
		# pd = np.array(Image.fromarray(pd))#.resize((512,256),Image.BICUBIC))
		# gt = np.array(Image.open(gts[15*i+j]))[:,512:,:3]
		# gt = np.hstack([gt[:,128:], gt[:,:128]])

		# res.append(Image.fromarray(np.hstack([gt, pd])))
		# continue

		pd_gray = color.rgb2yuv(pd)[..., 0] # 0~1
		gt_gray = color.rgb2yuv(gt)[..., 0] # 0~1

		pd_grad = (grad(pd_gray) + 1) / 2 # (-1~1) to (0~1)
		gt_grad = (grad(gt_gray) + 1) / 2 # (-1~1) to (0~1)

		psnr_val, rgb_val = sess.run([psnr, ssim_rgb], feed_dict={im1: gt[np.newaxis], im2: pd[np.newaxis]})
		yuv_val = sess.run(ssim_yuv, feed_dict={im3: gt_gray[np.newaxis,...,np.newaxis], im4: pd_gray[np.newaxis,...,np.newaxis]})
		multi_val = sess.run(ssim_multi, feed_dict={im3: gt_gray[np.newaxis,...,np.newaxis], im4: pd_gray[np.newaxis,...,np.newaxis]})
		sharp_val = sess.run(sharp, feed_dict={im3: gt_grad[np.newaxis], im4: pd_grad[np.newaxis]})

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
			i,
			'%.3lf' % np.mean(li_psnr),
			'%.3lf' % np.mean(li_ssim_rgb),
			'%.3lf' % np.mean(li_ssim_yuv),
			'%.3lf' % np.mean(li_ssim_multi),
			'%.3lf' % np.mean(li_sharp),
			'%.3lf' % np.mean(li_alx),
			'%.3lf' % np.mean(li_sqz),
			'%.3lf' % np.mean(li_vgg),
		)

	print(
		'center', i,
		'%.3lf' % np.mean(li_psnr[7::7]),
		'%.3lf' % np.mean(li_ssim_rgb[7::7]),
		'%.3lf' % np.mean(li_ssim_yuv[7::7]),
		'%.3lf' % np.mean(li_ssim_multi[7::7]),
		'%.3lf' % np.mean(li_sharp[7::7]),
		'%.3lf' % np.mean(li_alx[7::7]),
		'%.3lf' % np.mean(li_sqz[7::7]),
		'%.3lf' % np.mean(li_vgg[7::7]),
	)
	
	# res[0].save(f'{i}.gif', append_images=res[1:], save_all=True, duaration=150, loop=0)
	# res[0].save(f'{i}.png')
	# input('press')

