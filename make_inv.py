from PIL import Image
import numpy as np
import os, pickle, tqdm
import matplotlib.pyplot

def right_shift(pil_img, start, num_turn):
    assert(0 <= start and start < num_turn)
    img = np.array(pil_img)
    h, w = img.shape[:2]
    unit = w // num_turn
    assert((num_turn * unit) == w)
    img = np.hstack([img[:,(start*unit):], img[:,:(start*unit)]])
    return Image.fromarray(img)

def right_shift_unproj(pkl_unproj, start, num_turn):
    assert(0 <= start and start < num_turn)
    output, w, h = {}, 1024, 512
    for k in range(6):
        img = np.array(pkl_unproj[f'w{w}xh{h}'][2::3]).astype(np.int64).reshape((h, w))
        unit = w // num_turn
        assert((num_turn * unit) == w)
        img = np.hstack([img[:,(start*unit):], img[:,:(start*unit)]])
        x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        output[f'w{w}xh{h}'] = np.stack([y,x,img],axis=2).astype(np.int64).flatten().tolist()
        w //= 2
        h //= 2
    return output

if __name__ == '__main__':

    num_turn = 32
    vids = ['000','003','004','005','015','016','018','025','032','033','035','074','077','078','081','082','083','084','086','088','103','110','114','117','158','162','163','165','176','178','199','200','202','203','212','214','217','223','228','240','257','261','267','279','280','284','285','287','289','290']
    vids = [int(vid) for vid in vids]

    for i in tqdm.tqdm(vids):
        pre = 'london_vid2vid_xiaohu_test'
        for prefix, ext in [(f'{pre}/images', '.jpg'), (f'{pre}/seg_maps', '.png')]:
            final_imgs = []
            for j in range(2,17):
                img = Image.open(f'{prefix}/stuttgart_00_{i:03d}/stuttgart_00_000000_{j:06d}_leftImg8bit{ext}')
                final_imgs.append(img)


            for j in range(15):
                final_imgs.append(right_shift(final_imgs[14-j], 16, num_turn))

            final_imgs = final_imgs[-15:]
            final_imgs = final_imgs[0:1] + final_imgs
            final_imgs = final_imgs[0:1] + final_imgs
            for j, final_img in enumerate(final_imgs):
                new_pre = prefix.replace(pre, f'{pre}inv')
                os.makedirs(f'{new_pre}/stuttgart_00_{i:03d}', exist_ok=True)
                final_img.save(f'{new_pre}/stuttgart_00_{i:03d}/stuttgart_00_000000_{j:06d}_leftImg8bit{ext}')



        for prefix, ext in [(f'{pre}/unprojections', '.pkl')]:
            final_imgs = []
            for j in range(2,17):
                with open(f'{prefix}/stuttgart_00_{i:03d}/stuttgart_00_000000_{j:06d}_leftImg8bit{ext}', 'rb') as f:
                    final_imgs.append(pickle.load(f))

            for j in range(15):
                final_imgs.append(right_shift_unproj(final_imgs[14-j], 16, num_turn))

            final_imgs = final_imgs[-15:]
            final_imgs = final_imgs[0:1] + final_imgs
            final_imgs = final_imgs[0:1] + final_imgs
            for j, final_img in enumerate(final_imgs):
                new_pre = prefix.replace(pre, f'{pre}inv')
                os.makedirs(f'{new_pre}/stuttgart_00_{i:03d}', exist_ok=True)
                pickle.dump(final_img, open(f'{new_pre}/stuttgart_00_{i:03d}/stuttgart_00_000000_{j:06d}_leftImg8bit{ext}', 'wb'))

