import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random
import cv2 as cv

# 计算模型参数量
def model_param_count(model):
    return sum([p.numel() for p in model.parameters()])

# 峰值信噪比
def psnr(img1, img2, max_v=255):
    mse = ((img1-img2)**2).mean()
    if mse == 0:
        return float('inf')
    else:
        return 10*np.log10((max_v**2)/mse)

# 结构相似度
def ssim(img1, img2,L=255,K1=0.01,K2=0.03):
    C1,C2 = (K1*L)**2,(K2*L)**2
    miu_x,miu_y = img1.mean(),img2.mean()
    theta_x,theta_y = img1.std(),img2.std()
    theta_xy = np.cov(img1.flatten(),img2.flatten())[0,1]    
    ssim = ((2*miu_x*miu_y+C1)*(2*theta_xy+C2))/((miu_x**2+miu_y**2+C1)*(theta_x**2+theta_y**2+C2))
    return ssim

# 梯度幅相似性偏差
def gmsd(ref_img,dis_img,c=170,device='cuda'):
    # 输入类型检查
    if type(dis_img) == np.ndarray:
        assert dis_img.ndim == 2 or dis_img.ndim == 3
        if dis_img.ndim == 2:
            dis_img = torch.from_numpy(dis_img).unsqueeze(0).unsqueeze(0)
        else:
            dis_img = torch.from_numpy(dis_img).unsqueeze(0)
    if type(ref_img) == np.ndarray:
        assert ref_img.ndim == 2 or ref_img.ndim == 3
        if ref_img.ndim == 2:
            ref_img = torch.from_numpy(ref_img).unsqueeze(0).unsqueeze(0)
        else:
            ref_img = torch.from_numpy(ref_img).unsqueeze(0)
    # 算法需要输入为灰度图像，像素值0-255
    if torch.max(dis_img) <= 1:
        dis_img = dis_img * 255
    if torch.max(ref_img) <= 1:
        ref_img = ref_img * 255

    '''算法主体'''
    hx=torch.tensor([[1/3,0,-1/3]]*3,dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)#Prewitt算子
    ave_filter=torch.tensor([[0.25,0.25],[0.25,0.25]],dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)#均值滤波核
    down_step=2#下采样间隔
    hy=hx.transpose(2,3)

    dis_img=dis_img.float().to(device)
    ref_img=ref_img.float().to(device)

    #均值滤波
    ave_dis=F.conv2d(dis_img,ave_filter,stride=1)
    ave_ref=F.conv2d(ref_img,ave_filter,stride=1)
    #下采样
    ave_dis_down=ave_dis[:,:,0::down_step,0::down_step]
    ave_ref_down=ave_ref[:,:,0::down_step,0::down_step]
    #计算mr md等中间变量
    mr_sq=F.conv2d(ave_ref_down,hx)**2+F.conv2d(ave_ref_down,hy)**2
    md_sq=F.conv2d(ave_dis_down,hx)**2+F.conv2d(ave_dis_down,hy)**2
    mr=torch.sqrt(mr_sq)
    md=torch.sqrt(md_sq)
    GMS=(2*mr*md+c)/(mr_sq+md_sq+c)
    GMSD=torch.std(GMS.view(-1))
    return GMSD



# 均方根误差
def rmse(img1,img2):
    return ((img1-img2)**2).mean()**0.5
    

# 显示图片
def imshow(img_list,xyhw=None,cmap="gray"):
    img_numbers = len(img_list)
    plt.figure(figsize=(6*img_numbers,6*2))
    for i,img in enumerate(img_list):
        plt.subplot(2,img_numbers,i+1)
        plt.imshow(img,cmap)
    if xyhw is not None:
        x,y,h,w = xyhw
        for i,img in enumerate(img_list):
            plt.subplot(2,img_numbers,img_numbers+i+1)
            plt.axis("off")
            plt.imshow(img[x:x+h,y:y+w],cmap)
    plt.show()


# min-max归一化
def min_max_norm(x):
    if x.max()-x.min()==0:
        return np.zeros_like(x)
    else:
        return (x-x.min())/(x.max()-x.min())

# 傅里叶变换频谱
def fft(img):
    fft = np.fft.fft2(img) 
    fft = np.fft.fftshift(fft) 
    fft = 20*np.log(np.abs(fft))
    return fft 

# 大津分割 输入[0,1] 输出0，1Mask
def otsu(img):
    if not img.max()>1:
        img = img*255
    t,img_otsu = cv.threshold(img.astype(np.uint8),0,1,cv.THRESH_OTSU)
    return img_otsu


# 图像缓存，CycleGAN训练技巧，使用前几轮G生成的图像训练D
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (
            max_size > 0
        ), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# 学习率衰减策略
class LambdaLR:
    def __init__(
        self,
        n_epochs,  # 总训练轮数
        offset=0,  # 初始化为第几轮
        decay_start_epoch=100,  # 衰减开始轮次
    ):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        # 学习率衰减跟新策略：lr * (1 - 当前epoch / 总epoches)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )

# 训练信息管理
class TrainingInfo:
    def __init__(self,names) -> None:
        self.names = names
        self.dic = {k:[] for k in names}

    def append_key(self,k):
        self.names.append(k)
        self.dic[k]=[]

    def append_value(self,k,v):
        self.dic[k].append(v)        

    def get_k_mean(self,k):
        return np.mean(self.dic[k])

    def get_mean_info(self):
        info = ""
        for k in self.names:
            info+=f"{k}:{np.mean(self.dic[k])}\t"
        return info
    
    def clean(self):
        for k in self.names:
            self.dic[k] = []

