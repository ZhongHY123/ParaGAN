import numpy as np
import matplotlib.pyplot as plt
def read_npy():
    # path = r'TrainedModels/delta/2020_10_13_21_06_33_generation_train_depth_3_lr_scale_0.1_act_lrelu_0.05/gen_samples_stage_0/gen_sample_1.npy'
    path = r'gen_samples_stage_5/gen_sample_6.npy'
    path1 = r'0/real_scale.npy'
    x = np.load(path)
    f = open(r'D:\PythonProject\paraGAN\SGEMS\delta2.txt','w+')
    f.write(str(x.shape)+'\n')
    f.write('1\nfacies\n')
    for i in range(x.shape[0]):
        for m in range(x.shape[1]):
            for n in range(x.shape[2]):
                if x[i,m,n]<=0.5:
                    f.write('0\n')
                elif x[i,m,n]>0.5 and x[i,m,n]<1.5:
                    f.write('1\n')
                else:
                    f.write('2\n')

    print(x.shape)
    print(x)
    plt.imshow(x[0])
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    read_npy()