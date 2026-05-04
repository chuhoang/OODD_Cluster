import os
'''
path="./data/images_classic/cinic/valid"
save_path="./data/benchmark_imglist/cifar10/val_cinic10.txt"
prefix="cinic/valid/"
category=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
with open(save_path,'a') as f:
    for name in category:
        label=category.index(name)
        sub_path=path+'/'+name
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+name+'/'+file+' '+str(label)+'\n'
            f.write(line)
    f.close()       
'''

path="/disk1/yangyifeng/icml_2024/OpenOOD/data/images_largescale/imagenet_1k/val"
save_path="/disk1/yangyifeng/icml_2024/OpenOOD/data/benchmark_imglist/imagenet/test_imagenet.txt"
prefix="cifar100c/"
files=os.listdir(path)
# 把valprep.sh删除
files.remove("valprep.sh")
# 按sort排序
files.sort()
with open(save_path,'a') as f:
    label=0
    for file in files:
        splits=file.split("_")
        pic_list = os.listdir(path+'/'+file)
        for pic in pic_list:
            line="imagenet_1k/val/"+file+'/'+pic+" "+str(label)+'\n'
            f.write(line)
        label = label + 1
    f.close()  

'''
path="./data/images_largescale/imagenet_v2"
save_path="./data/benchmark_imglist/imagenet/test_imagenetv2.txt"
prefix="imagenet_v2/"
with open(save_path,'a') as f:
    for i in range(0,1000):
        label=str(i)
        sub_path=path+'/'+label
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+label+'/'+file+' '+label+'\n'
            f.write(line)
    f.close() 
'''