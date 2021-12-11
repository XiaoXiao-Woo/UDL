import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import psutil
from multiprocessing import Pool
import cv2
path = "../../path_to_imagenet/val"
errors_file = []
# count = 0

LOG_FOUT2 = open("ok.txt", 'w')
def log_string2(out_str):
     LOG_FOUT2.write(out_str+'\n')
     LOG_FOUT2.flush()
     print(out_str)

LOG_FOUT = open("errors.txt", 'w')
def log_string(out_str):
     LOG_FOUT.write(out_str+'\n')
     LOG_FOUT.flush()
     print(out_str)




def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))

idx = 0

# def scan(dirs):
#         print("loading... : ", dirs, show_memory_info("to_image.py"))
#         abs_dirs = os.path.join(path, dirs)
#         try:
#             for files in os.listdir(abs_dirs):
#                 try:
#                     abs_file = os.path.join(abs_dirs, files)
#                     with open(abs_file, "rb") as f:
#                         img = Image.open(f)
#                         gc.collect()
#                 except:
#                     print("error file:", abs_file)
#                     errors_file.append(abs_file)
#                     continue
#         except:
#             errors_file.append(abs_file)
#             print("error file:", abs_file)
#     # count += 1
#     # if count == 2:
#     #     print(errors_file)
#     #     break
# for i, dirs in enumerate(os.listdir(path)):
#     if 'n03874599' == dirs:
#         idx = i
#         break
# files = os.listdir(path)[idx:]
# # print(files)
# pool = Pool(4)
# pool.map(scan, files)
# pool.close()
# pool.join()
#n03884397
# for i, dirs in enumerate(os.listdir(path)):
#     #if 'n03937543' == dirs:
#     if 'n04296562' == dirs:
#         idx = i
#         break
files = os.listdir(path)#[idx:]
def scan_picture(files):
    for ix, dirs in enumerate(files):#os.listdir(path)):
        if ix < idx:
            continue
        log_string2("loading... : " + dirs)
        abs_dirs = os.path.join(path, dirs)
        print("[{}/{}]".format(ix, len(files)))
        try:
            for file in os.listdir(abs_dirs):
                try:
                    abs_file = os.path.join(abs_dirs, file)
                    # with open(abs_file, "rb") as f:
                        # img = Image.open(f)
                    img = cv2.imread(abs_file)
                except:#图片损坏
                    # print("open error file:", abs_file)
                    # log_string("\t open error file:" + abs_file)
                    print(img)
                    log_string("open error file:" + abs_dirs + "\t\t/" + file)
                    errors_file.append(abs_dirs)
                    break
        except:
            # 空文件夹
            errors_file.append(abs_dirs)
            log_string("open error list_dirs:" + abs_dirs)
            # print("error list_dirs:", abs_dirs)
            continue
    LOG_FOUT.close()
    LOG_FOUT2.close()



# def pbar_show(files):
#     ix = -1
#     for dirs in tqdm.tqdm(range(len(files)), total=len(files), desc="scan:"):#os.listdir(path)):
#         ix += 1
#         if ix < idx:
#             continue
#         print("loading... : ", dirs, show_memory_info("to_image.py"))
#         abs_dirs = os.path.join(path, dirs)
#
#         try:
#             for files in os.listdir(abs_dirs):
#                 try:
#                     abs_file = os.path.join(abs_dirs, files)
#                     # with open(abs_file, "rb") as f:
#                         # img = Image.open(f)
#                     img = cv2.imread(abs_file,)
#                     log_string2(abs_file)
#                 except:#图片损坏
#                     # print("open error file:", abs_file)
#                     # log_string("\t open error file:" + abs_file)
#                     print(img)
#                     log_string("open error file:" + abs_dirs + "\t\t/" + files)
#                     errors_file.append(abs_dirs)
#                     break
#         except:
#             # 空文件夹
#             errors_file.append(abs_dirs)
#             log_string("open error list_dirs:" + abs_dirs)
#             # print("error list_dirs:", abs_dirs)
#             continue
#     LOG_FOUT.close()
#     LOG_FOUT2.close()

if __name__ == "__main__":
    # from multiprocessing import Pool
    # print(files)
    # with Pool(4) as p:
    #     p.map(scan_picture, files)
    print(files)
    scan_picture(files)









# print(errors_file)
# with open("errors.txt", "wb") as f:
#     for line in errors_file:
#         f.write(line + "\n")
#         f.flush()
        # f.write(line.encode('utf-8'))
        # f.write(u"\n")
#error list_dirs: ../../path_to_imagenet/train/n03937543
#error file: ../../path_to_imagenet/train/n03992509/n03992509_12927.JPEG
#error file: ../../path_to_imagenet/train/n03884397/n03884397_9968.JPEG
#error file: ../../path_to_imagenet/train/n04146614/n04146614_12958.JPEG