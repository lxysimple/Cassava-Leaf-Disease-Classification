'''
Fmix paper from arxiv: https://arxiv.org/abs/2002.12047
Fmix code from github : https://github.com/ecs-vlc/FMix
'''
import math
import random
import numpy as np
from scipy.stats import beta


def fftfreqnd(h, w=None, z=None):
    """求出频域 F的一个参数:频率

    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ 生成过滤掉高频,随机高斯分布的z

    Args:
        :param freqs: Bin values for the discrete fourier transform,过滤高频所需的一个超参数
        :param decay_power: Decay power for frequency decay prop 1/f**d,过滤高频所需的零一个超参数,控制高频阙值
        :param ch: Number of channels for the resulting mask,z的通道
        :param h: Required, first dimension size,z的高
        :param w: Optional, second dimension size,z的宽
        :param z: Optional, third dimension size,z也能是立体的

    Returns:
        (ch,h,w,z=0):一个基于高斯分布采样出的矩阵z,并且过滤掉了高频

    """
    # 通过×一个分数来对z所有数进行缩小,从而过滤掉高频,而scale就是这个缩小分数
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    # 从高斯分布上随机采样出z
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]
    #相乘过滤掉高频
    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    """ 得到fmix的裁剪矩阵mask

    Args:
        :param decay_power: Decay power for frequency decay prop 1/f**d
        :param shape: Shape of desired mask, list up to 3 dims
        :param ch: Number of channels for desired mask

    Returns:
        (1,w,h): 返回二值化前的mask矩阵
    """

    freqs = fftfreqnd(*shape)
    # 获得随机采样出、过滤掉高频的z
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    # 将z的实部维度与虚部维度相加,得到真正的z
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    # 将z进行傅里叶逆变换,取出实数部分,得到最后的mask,shape=(1,w,h)
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    # 对mask进行归一化,归一化到(0,1)之间
    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, reformulate=False):
    """ 从β分布中抽样出lam

    Args:
        :param alpha: Alpha value for beta distribution
        :param reformulate: If True, uses the reformulation of [1].

    """
    if reformulate:
        lam = beta.rvs(alpha+1, alpha) # rvs(arg1,arg2,loc=期望, scale=标准差, size=生成随机数的个数) 从分布中生成指定个数的随机数
    else:
        lam = beta.rvs(alpha, alpha)   # rvs(arg1,arg2,loc=期望, scale=标准差, size=生成随机数的个数) 从分布中生成指定个数的随机数

    return lam


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ 对mask进行二值化处理

    Args:
        :param mask: Low frequency image, usually the result of `make_low_freq_image`
        lam (int): Mean value of final mask,我感觉是二值化的阙值
        :param in_shape: Shape of inputs
        param max_soft (float): 让0-1边界变得很柔和,Softening value between 0 and 0.5 which smooths hard edges in the mask.
        :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask


def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ 得到二值化后的mask,Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda

    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda,二值化的阙值
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    # 二值化mask,是否羽化由max_soft决定
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask


def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ 随机抽取同一batch的图片,和原图进行fmix

    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])

    x1, x2 = x * mask, x[index] * (1-mask)
    return x1+x2, index, lam


class FMixBase:
    """ FMix augmentation

        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
    """

    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__()
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.size = size
        self.alpha = alpha
        self.max_soft = max_soft
        self.index = None
        self.lam = None

    def __call__(self, x):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError
