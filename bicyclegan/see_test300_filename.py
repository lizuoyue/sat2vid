import os, glob
files = sorted(glob.glob('london_seltest_15_frames/*_00.png'))
# print(os.path.basename(files[178]).replace('_00.png', ''))
# print(os.path.basename(files[86]).replace('_00.png', ''))
for file in files:
    print(os.path.basename(file).replace('_00.png', ''))
