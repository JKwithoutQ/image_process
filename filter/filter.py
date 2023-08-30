import numpy as np


def conv_2d_single_kernel(input_data, kernel, stride):
    """单个卷积核进行卷积，得到单个输出。
    由于是学习卷积实现原理这里简单处理，padding 是自动补全，
    相当于tf 里面的 "SAME"。
    Args:
        input_data: 卷积层输入，是一个 shape 为 [c, h, w]
            的 np.array。
        kernel: 卷积核大小，形式如 [c, k_h, k_w] ，c 为输
            入 input_data 的通道数。
        stride: stride， list [s_h, s_w]。
    Return:
        out: 卷积结果
    Raises:
        ValueError: input_data 的 channels 与 kernel 的
            c 不相等，会抛异常。
    """
    c, h, w = input_data.shape
    kernel_c, kernel_h, kernel_w = kernel.shape
    if c != kernel_c:
        raise ValueError("channels: input_data:{}, kernel:{}".format(c, kernel_c))

    stride_h, stride_w = stride

    padding_h = (kernel_h - 1) // 2
    padding_w = (kernel_w - 1) // 2
    padding_data = np.zeros((c, h + padding_h * 2, w + padding_w * 2))
    padding_data[:, padding_h:-padding_h, padding_w:-padding_w] = input_data

    out = np.zeros((h // stride_h, w // stride_w))
    for idx_h, i in enumerate(range(0, h - kernel_h, stride_h)):
        for idx_w, j in enumerate(range(0, w - kernel_w, stride_w)):
            window = padding_data[:, i:i + kernel_h, j:j + kernel_w]
            out[idx_h, idx_w] = np.sum(window * kernel)
    return out


def conv2d(input_data, kernel, stride=1):
    """多个卷积核计算卷积，生成多通道输出。
    Args:
        input_data: 卷积层输入，是一个 shape 为 [c, h, w]
            的 np.array。
        kernel: 卷积核大小，形式如 [out_c, c, k_h, k_w]
            ，c 为输入 input_data 的通道数， out_c 为卷
            积核个数，也是输出数据的通道数。
        stride: stride， 形式如 [h, w] 或 s。
    Return:
        out: 卷积结果
    """
    h, w = input_data.shape[1:3]
    out_c = kernel.shape[0]

    if isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    else:
        stride_h, stride_w = stride

    out = np.zeros((out_c, h // stride_h, w // stride_w))
    for n in range(out_c):
        out[n, :, :] = conv_2d_single_kernel(input_data, kernel[n, :, :], (stride_h, stride_w))
    return out
