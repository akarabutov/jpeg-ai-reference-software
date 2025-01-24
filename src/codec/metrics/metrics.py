# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from src.codec.common import Image
from typing import List


class DataClass:
    def __init__(self):
        self.data_range = [0, 1]
        self.yuv_data = {}
        self.rgb_data = None
        self.yuv_format = '444'
        self.shape = []  # height, width
        self.bitdepth = -1

    def load_image(self, filename, def_bits=10, def_fmt='444', device='cuda', color_conv='709'):
        ext = filename[-4:].lower()
        bitdepth_ans = def_fmt
        if ext == '.yuv':
            w, h, b, fmt = DataClass.extract_info(filename, def_bits, def_fmt)
            if def_bits == -1:
                def_bits = b
            assert def_bits > 0
            self.shape = [h, w]
            yuv_data = DataClass.read_yuv(filename,
                                          w,
                                          h,
                                          b,
                                          fmt=fmt,
                                          device=device,
                                          out_plane_norm=self.data_range)
            self.yuv_format = fmt

            for plane in yuv_data:
                self.yuv_data[plane] = DataClass.round_plane(yuv_data[plane], def_bits)

            yuv_t = self.convert_yuvdict_to_tensor(self.yuv_data)
            self.rgb_data = DataClass.yuv_to_rgb(yuv_t, color_conv)
            # Should we round RGB to n bits?
            self.rgb_data = DataClass.round_plane(self.rgb_data, def_bits)
            bitdepth_ans = def_bits
        elif ext == '.png' or ext == '.jpg' or ext == '.JPEG' or ext == 'jpeg':
            bgr_data = cv2.imread(filename,  cv2.IMREAD_UNCHANGED)
            rgb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)
            if def_bits == -1:
                def_bits = 8 if rgb_data.dtype == 'uint8' else 16
            max_val = (1 << (8 if rgb_data.dtype == 'uint8' else 16)) - 1
            self.rgb_data = torch.tensor(rgb_data.astype(dtype=np.int32), dtype=torch.float,
                                         device=device).permute(2, 0, 1)
            self.rgb_data = DataClass.convert_and_round_plane(self.rgb_data, [0, max_val],
                                                              self.data_range,
                                                              def_bits).unsqueeze(0)
            yuv_t = self.rgb_to_yuv(self.rgb_data, color_conv).clamp(min(self.data_range),
                                                                     max(self.data_range))
            yuv_t = DataClass.round_plane(yuv_t, def_bits)
            self.yuv_format = '444'
            self.shape = yuv_t.shape[-2:]
            bitdepth_ans = def_bits
            self.yuv_data = {'Y': yuv_t[0, 0], 'U': yuv_t[0, 1], 'V': yuv_t[0, 2]}
        else:
            raise NotImplementedError
        self.bitdepth = bitdepth_ans
        return self, bitdepth_ans

    @staticmethod
    def from_ImageCls(image: Image):
        data = DataClass()
        data.data_range = [0, 1]

        image.convert_range_(data.data_range)
        data.yuv_format = image.format
        if image.is_YUV():
            data.yuv_data = {
                "Y": image.get_component("a")[0, 0, :, :].clone(),
                "U": image.get_component("b")[0, 0, :, :].clone(),
                "V": image.get_component("c")[0, 0, :, :].clone(),
            }
            image.to_444_()
            image.to_RGB_()
            data.rgb_data = image.get_tensor().clone()
        else:
            data.rgb_data = image.get_tensor().clone()
            image.to_YUV_()
            data.yuv_data = {
                "Y": image.get_component("a")[0, 0, :, :].clone(),
                "U": image.get_component("b")[0, 0, :, :].clone(),
                "V": image.get_component("c")[0, 0, :, :].clone(),
            }
        data.shape = image.shape[-2:]
        data.bitdepth = image.bit_depth

        return data

    @staticmethod
    def round_plane(plane, bits):
        return plane.mul((1 << bits) - 1).round().div((1 << bits) - 1)

    @staticmethod
    def convertup_and_round_plane(plane, cur_range, new_range, bits):
        return DataClass.convert_range(plane, cur_range, new_range).mul((1 << bits) - 1).round()

    @staticmethod
    def convert_and_round_plane(plane, cur_range, new_range, bits):
        return DataClass.round_plane(DataClass.convert_range(plane, cur_range, new_range), bits)

    @staticmethod
    def convert_range(plane, cur_range, new_range=[0, 1]):
        if cur_range[0] == new_range[0] and cur_range[1] == new_range[1]:
            return plane
        return (plane + cur_range[0]) * (new_range[1] - new_range[0]) / (
            cur_range[1] - cur_range[0]) - new_range[0]

    @staticmethod
    def convert_yuvdict_to_tensor(yuv, device='cuda'):
        size = yuv['Y'].shape
        c = len(yuv)
        ans = torch.zeros((1, c, size[-2], size[-1]),
                          dtype=torch.float,
                          device=torch.device(device))
        ans[:, 0, :, :] = yuv['Y']
        ans[:, 1, :, :] = yuv['U'] if 'U' in yuv else yuv['Y']
        ans[:, 2, :, :] = yuv['V'] if 'V' in yuv else yuv['Y']
        return ans

    @staticmethod
    def color_conv_matrix(color_conv='709'):
        if color_conv == '601':
            # BT.601
            a = 0.299
            b = 0.587
            c = 0.114
            d = 1.772
            e = 1.402
        elif color_conv == '709':
            # BT.709
            a = 0.2126
            b = 0.7152
            c = 0.0722
            d = 1.8556
            e = 1.5748
        elif color_conv == '2020':
            # BT.2020
            a = 0.2627
            b = 0.6780
            c = 0.0593
            d = 1.8814
            e = 1.4747
        else:
            raise NotImplementedError

        return a, b, c, d, e

    @staticmethod
    def yuv_to_rgb(image: torch.Tensor, color_conv='709') -> torch.Tensor:
        r"""Convert an YUV image to RGB.

        The image data is assumed to be in the range of (0, 1).

        Args:
            image (torch.Tensor): YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

        Returns:
            torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

        Example:
            >>> input = torch.rand(2, 3, 4, 5)
            >>> output = yuv_to_rgb(input)  # 2x3x4x5

        Took from https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html#rgb_to_yuv
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(
                image.shape))

        y: torch.Tensor = image[..., 0, :, :]
        u: torch.Tensor = image[..., 1, :, :] - 0.5
        v: torch.Tensor = image[..., 2, :, :] - 0.5

        a, b, c, d, e = DataClass.color_conv_matrix(color_conv)

        r: torch.Tensor = y + e * v  # coefficient for g is 0
        g: torch.Tensor = y - (c * d / b) * u - (a * e / b) * v
        b: torch.Tensor = y + d * u  # coefficient for b is 0

        out: torch.Tensor = torch.stack([r, g, b], -3)

        return out

    @staticmethod
    def rgb_to_yuv(image: torch.Tensor, color_conv='709') -> torch.Tensor:
        r"""Convert an RGB image to YUV.

        The image data is assumed to be in the range of (0, 1).

        Args:
            image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

        Returns:
            torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

        Example:
            >>> input = torch.rand(2, 3, 4, 5)
            >>> output = rgb_to_yuv(input)  # 2x3x4x5
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(
                image.shape))

        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        a1, b1, c1, d1, e1 = DataClass.color_conv_matrix(color_conv)

        y: torch.Tensor = a1 * r + b1 * g + c1 * b
        u: torch.Tensor = (b - y) / d1 + 0.5
        v: torch.Tensor = (r - y) / e1 + 0.5

        out: torch.Tensor = torch.stack([y, u, v], -3)

        return out

    @staticmethod
    def extract_info(fn, default_bits=10, default_fmt='444'):
        import re
        wh = re.search(r'(?P<w>\d+)x(?P<h>\d+)', fn)
        b = re.search(r'(?P<b>\d+)bit', fn)

        w = wh.group('w')
        h = wh.group('h')

        b = default_bits if b is None else b.group('b')
        fmt = default_fmt
        if "YUV444" in fn:
            fmt = "444"
        elif "YUV420" in fn:
            fmt = "420"
            # raise NotImplementedError
            # TODO: add upsampling in YUV -> RGB convertion
        elif 'sRGB' in fn:
            fmt = 'sRGB'

        return int(w), int(h), int(b), fmt

    @staticmethod
    def read_yuv(filename, width, height, bits=8, out_plane_norm=[0, 1], fmt='444', device='cuda'):
        nr_bytes = int(np.ceil(bits / 8))
        if nr_bytes == 1:
            data_type = np.uint8
        elif nr_bytes == 2:
            data_type = np.uint16
        else:
            raise NotImplementedError('Reading more than 16-bits is currently not supported!')

        ans = {'Y': None, 'U': None, 'V': None}
        sizes = {'Y': [height, width], 'U': [height, width], 'V': [height, width]}

        if fmt == '420':
            for a in ['U', 'V']:
                sizes[a][0] >>= 1
                sizes[a][1] >>= 1
        elif fmt == '400':
            ans = {'Y': None}
            for a in ['U', 'V']:
                sizes[a][0] = 0
                sizes[a][1] = 0
        elif fmt == '444':
            pass
        else:
            raise NotImplementedError('The specified yuv format is not supported!')

        for plane in ans:
            ans[plane] = torch.zeros(sizes[plane], dtype=torch.float, device=torch.device(device))

        with open(filename, 'rb') as f:
            for plane in ['Y', 'U', 'V']:
                size = np.int(sizes[plane][0] * sizes[plane][1] * nr_bytes)
                tmp = np.frombuffer(f.read(size), dtype=data_type)
                tmp = tmp.reshape(sizes[plane])

                ans[plane] = torch.tensor(
                    (tmp.astype(np.float32) / (2**bits - 1)) *
                    (max(out_plane_norm) - min(out_plane_norm)) + min(out_plane_norm),
                    dtype=torch.float,
                    device=torch.device(device))

        return ans

    def write_yuv(self, f, bits=None):
        """
        dump a yuv file to the provided path
        @path: path to dump yuv to (file must exist)
        @bits: bitdepth
        @frame_idx: at which idx to write the frame (replace), -1 to append
        """
        if bits is None:
            bits = self.bitdepth
        yuv = self.yuv_data.copy()
        nr_bytes = np.ceil(bits / 8)
        if nr_bytes == 1:
            data_type = np.uint8
        elif nr_bytes == 2:
            data_type = np.uint16
        elif nr_bytes <= 4:
            data_type = np.uint32
        else:
            raise NotImplementedError('Writing more than 16-bits is currently not supported!')

        # rescale to range of bits
        for plane in yuv:
            yuv[plane] = DataClass.convertup_and_round_plane(yuv[plane], self.data_range,
                                                             self.data_range, bits).cpu().numpy()

        # dump to file
        lst = []
        for plane in ['Y', 'U', 'V']:
            if plane in yuv.keys():
                lst = lst + yuv[plane].ravel().tolist()

        raw = np.array(lst)

        raw.astype(data_type).tofile(f)


class MetricParent:
    def __init__(self, bits=10, max_val=1023, mvn=1, name='', device=torch.device('cuda')):
        self.__name = name
        self.bits = bits
        self.max_val = max_val
        self.__metric_val_number = mvn
        self.metric_name = ''
        self.device = device

    def set_bd_n_maxval(self, bitdepth=None, max_val=None):
        if bitdepth is not None:
            self.bits = bitdepth
        if max_val is not None:
            self.max_val = max_val

    def name(self):
        return self.__name

    def metric_val_number(self):
        return self.__metric_val_number

    def calc(self, orig, rec):
        raise NotImplementedError

    def pad_tensor_till_size(self, tensor, min_size):
        s = tensor.shape[-2:]
        ph = max(min_size-s[0], 0)
        pw = max(min_size-s[1], 0)
        return F.pad(tensor, (0, pw, 0, ph), 'constant', 0)


class PSNRMetric(MetricParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, mvn=3, name=['PSNR_Y', 'PSNR_U', 'PSNR_V'])

    def calc(self, orig, rec):
        ans = []
        for plane in orig.yuv_data:
            a = orig.yuv_data[plane].mul((1 << self.bits) - 1)
            b = rec.yuv_data[plane].mul((1 << self.bits) - 1)
            mse = torch.mean((a - b)**2).item()
            if mse == 0.0:
                ans.append(100)
            else:
                ans.append(20 * np.log10(self.max_val) - 10 * np.log10(mse))
        return ans


class MSSSIMTorch(MetricParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name='MS-SSIM (PyTorch)')

    def calc(self, orig, rec):
        ans = 0.0
        from pytorch_msssim import ms_ssim
        if 'Y' not in orig.yuv_data or 'Y' not in rec.yuv_data:
            return -100.0
        plane = 'Y'
        a = orig.yuv_data[plane].mul((1 << self.bits) - 1)
        b = rec.yuv_data[plane].mul((1 << self.bits) - 1)
        a.unsqueeze_(0).unsqueeze_(0)
        b.unsqueeze_(0).unsqueeze_(0)
        a = self.pad_tensor_till_size(a, 164)
        b = self.pad_tensor_till_size(b, 164)
        ans = ms_ssim(a, b, data_range=self.max_val).item()

        return ans


class MetricsFabric:
    metrics_list = [
        'msssim_torch', 'psnr'
    ]

    def __init__(self, bits={}, max_vals={}):
        self.bits = bits
        self.max_vals = max_vals

    def create_instance(self, name):
        name = name.lower()

        params = {}
        if isinstance(self.bits, dict):
            if name in self.bits:
                params['bits'] = self.bits[name]
            else:
                params['bits'] = 10
        else:
            params['bits'] = self.bits

        if isinstance(self.max_vals, dict):
            if name in self.max_vals:
                params['max_val'] = self.max_vals[name]
            else:
                params['max_val'] = (1 << params['bits']) - 1
        else:
            params['max_val'] = self.max_vals

        ans = None
        if name == 'msssim_torch':
            ans = MSSSIMTorch(**params)
        elif name == 'psnr':
            ans = PSNRMetric(**params)
        else:
            raise NotImplementedError

        ans.metric_name = name
        return ans


class MetricsProcessor:
    def __init__(self):
        self.internal_bits = -1
        self.jvet_psnr = False
        self.metrics = MetricsFabric.metrics_list
        self.metrics_output = MetricsFabric.metrics_list
        self.color_conv = '709'
        self.metrics_fab = MetricsFabric()
        self.metrics_inst = {}
        self.f = None
        self.sep = '\t'
        self.is_csv = False
        for m in MetricsFabric.metrics_list:
            self.metrics_inst[m] = self.metrics_fab.create_instance(m)
        self.additional_titles = []

    def add_titles(self, titles):
        """
        Add title(s). Should go before init_summary_file()

        Args:
            titles (str, list): title or titles
        """
        if isinstance(titles, list):
            self.additional_titles += titles
        elif isinstance(titles, str):
            self.additional_titles += [titles]
        else:
            raise ValueError

    def get_titles(self):
        """
        Get list with titles

        Returns:
            list: titles
        """
        titles = []
        titles.append('Reconstruct')
        titles.append('Original')
        titles.append('Codec')
        titles.append('BPP')
        for m_t in self.metrics_output:
            m = self.metrics_inst[m_t]
            name = m.name()
            if isinstance(name, list):
                titles += name
            else:
                titles.append(name)
        titles.append('MAC/pxl')
        titles.append('DecGPU')
        titles.append('DecCPU')
        titles.append('EncGPU')
        titles.append('EncCPU')
        return titles + self.additional_titles

    def add_arguments(self, ap):
        ap.add_argument('--internal-bits',
                        type=int,
                        default=10,
                        choices=[-1, 8, 10],
                        help=r'Bits for internal calculations (8,10 (default) or determined '
                        r'based on internal data representation of the input file )')
        ap.add_argument('--jvet-psnr',
                        default=False,
                        action='store_true',
                        help='Use 1020 as upper bound for 10 bits')
        ap.add_argument('--metrics',
                        default=MetricsFabric.metrics_list,
                        choices=MetricsFabric.metrics_list,
                        nargs='+',
                        help=r'Metrics to be used. Default: ['
                        f'{" ".join(MetricsFabric.metrics_list)}]')
        ap.add_argument('--metrics_output',
                        default=MetricsFabric.metrics_list,
                        choices=MetricsFabric.metrics_list,
                        nargs='+',
                        help=r'Order of metrics in output. Default: '
                        f"[{' '.join(MetricsFabric.metrics_list)}]")
        ap.add_argument('--color-conv',
                        default='709',
                        choices=['601', '709', '2020'],
                        help='Color convertion notation')
        ap.add_argument('--max_samples_for_eval_on_gpu',
                        type=int,
                        default=-1,
                        help='Maximum number of samples for gpu processing. Larger images are processed on CPU. If it is equal to -1, teh data won\'t move to CPU.')

    def parse_arguments(self, ap, cmd_args: List[str] = None):
        args, _ = ap.parse_known_args(args=cmd_args)
        self.internal_bits = args.internal_bits
        self.jvet_psnr = args.jvet_psnr
        self.metrics = args.metrics
        self.metrics_output = args.metrics_output
        self.color_conv = args.color_conv
        self.max_samples_for_eval_on_gpu = args.max_samples_for_eval_on_gpu
        return args

    def __del__(self):
        self.close_file()

    @staticmethod
    def compute_bpp(filename, shape):
        bs_size = os.path.getsize(filename)
        bpp = bs_size * 8 / (shape[-2] * shape[-1])
        return bpp

    def process_images(self, orig: DataClass, rec: DataClass):
        bits = orig.bitdepth
        if (self.max_samples_for_eval_on_gpu != -1) and (orig.yuv_data['Y'].numel() > self.max_samples_for_eval_on_gpu):
            # move data to cpu in order for metrics to run on cpu. avoding OOM on gpu
            # if image has more samples than max_samples_for_eval_on_gpu
            for data in [orig, rec]:
                for plane in data.yuv_data:
                    data.yuv_data[plane] = data.yuv_data[plane].cpu()
                data.rgb_data = data.rgb_data.cpu()
        ans = []
        for m_t in self.metrics_output:
            if m_t in self.metrics:
                m = self.metrics_inst[m_t]
                if m.metric_name == 'psnr' and self.jvet_psnr:
                    max_val = 255 * (1 << (bits - 8))
                else:
                    max_val = (1 << bits) - 1
                m.set_bd_n_maxval(bits, max_val)

                tmp = m.calc(orig, rec)
                if isinstance(tmp, list):
                    ans += tmp
                else:
                    ans.append(tmp)
            else:
                ans.append(-100)
        return ans

    def process_yuv_files(self, shape, orig_fn, rec_fn, device: str = "cpu"):
        w, h, b, fmt = Image.extract_info(orig_fn, default_bits=8)
        orig_img = Image.read_yuv(orig_fn, w, h, b, fmt=fmt, device=device)
        rec_img = Image.read_yuv(rec_fn, w, h, b, fmt=fmt, device=device)
        data_o = DataClass.from_ImageCls(orig_img)
        data_r = DataClass.from_ImageCls(rec_img)
        return self.process_images(data_o, data_r)

    def process_image_files(self, orig_fn: str, rec_fn: str, device: str = "cpu"):
        data_o, target_bd = DataClass().load_image(
            orig_fn,
            def_bits=self.internal_bits,
            color_conv=self.color_conv,
            device=device,
        )
        data_r, _ = DataClass().load_image(
            rec_fn, def_bits=target_bd, color_conv=self.color_conv, device=device
        )
        ans = self.process_images(data_o, data_r)  # , data_o.shape

        return ans

    def init_summary_file(self, fn, mode='txt'):
        self.f = open(fn, 'w')
        self.sep = '\t'
        if mode == 'csv':
            self.sep = ', '
            self.is_csv = True
            titles = self.get_titles()
            self.f.write(f'{self.sep.join(titles)}\n')

    @staticmethod
    def init_ptflops_calc(model):
        import sys

        from ptflops.flops_counter import add_flops_counting_methods
        ans = add_flops_counting_methods(model)
        ans.start_flops_count(ost=sys.stdout, verbose=False, ignore_list=[])
        ans.__batch_counter__ = 1
        return ans

    @staticmethod
    def reset_ptflops_variables(module):
        from ptflops.flops_counter import get_model_parameters_number
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


    @staticmethod
    def reset_ptflops_calc(model):
        import sys
        ans = model
        if not hasattr(model, 'start_flops_count'):
            from ptflops.flops_counter import add_flops_counting_methods
            ans = add_flops_counting_methods(model)
        else:
            MetricsProcessor.reset_ptflops_variables(model)
            model.apply(MetricsProcessor.reset_ptflops_variables)
        ans.start_flops_count(ost=sys.stdout, verbose=False, ignore_list=[])
        ans.__batch_counter__ = 1
        return ans

    @staticmethod
    def finish_ptflops_calc(model, size=None):
        """
        Get flops of the model

        Args:
            model (nn.Module): model, which we examine
            size (list, optional): shape of the input image (width, height). Defaults to None.

        Returns:
            float: MACs
            float: MACs per pixel
        """
        # Flops calculation
        flops_count, _ = model.compute_average_flops_cost()
        flops_per_pixel = None
        if size is not None:
            flops_per_pixel = flops_count / (size[0] * size[1])
        return flops_count, flops_per_pixel

    @staticmethod
    def store_complexity_info(rec_path,
                              kmac=None,
                              decGPU=None,
                              decCPU=None,
                              encGPU=None,
                              encCPU=None):
        """
        Store additional information

        Args:
            rec_path (str): path to reconstructed file
            kmac (float): complexity in kMAC per pixel
            decGPU (float): decoder complexity on GPU in sec.
            decCPU (float): decoder complexity on CPU in sec.
            encGPU (float): encoder complexity on GPU in sec.
            encCPU (float): encoder complexity on CPU in sec.
        """
        import json
        r_fn, _ = os.path.splitext(rec_path)
        out_path = f'{r_fn}.json'
        info_dict = {
            'kmac': kmac,
            'decGPU': decGPU,
            'decCPU': decCPU,
            'encGPU': encGPU,
            'encCPU': encCPU
        }
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                try:
                    cur_data = json.load(f)
                except:  # noqa: E722
                    cur_data = {}
        else:
            cur_data = {}
        cur_data.update(info_dict)
        with open(out_path, 'w') as f:
            json.dump(cur_data, f)

    @staticmethod
    def load_complexity_info(rec_path):
        """
        Load additional information

        Args:
            rec_path (str): path to reconstructed file
        Return:
            kmac (float): complexity in kMAC per pixel
            decGPU (float): decoder complexity on GPU in sec.
            decCPU (float): decoder complexity on CPU in sec.
            encGPU (float): encoder complexity on GPU in sec.
            encCPU (float): encoder complexity on CPU in sec.
        """
        import json
        r_fn, _ = os.path.splitext(rec_path)
        in_path = f'{r_fn}.json'
        ans = {'kmac': None, 'decGPU': None, 'decCPU': None, 'encGPU': None, 'encCPU': None}
        if os.path.exists(in_path):
            with open(in_path, 'r') as f:
                ans = json.load(f)
        return ans['kmac'], ans['decGPU'], ans['decCPU'], ans['encGPU'], ans['encCPU']

    def get_output_str(self,
                       seq_name,
                       bpp=None,
                       metrics=None,
                       prefix_data=None,
                       postfix_list=None):
        if bpp is None:
            bpp = -1
        if metrics is None:
            metrics = []
            for m_t in self.metrics_output:
                if m_t in self.metrics_inst:
                    m = self.metrics_inst[m_t]
                    for _ in range(m.metric_val_number()):
                        metrics.append(-1)

        a = [bpp] + metrics
        a = [str(x) for x in a]
        if self.is_csv and prefix_data is not None:
            if isinstance(prefix_data, list):
                a = prefix_data + a
            else:
                a = [prefix_data] + a
        if postfix_list is not None:
            for x in postfix_list:
                a.append('None' if x is None else str(x))
        a = [seq_name] + a
        return self.sep.join(a)

    def write_data(self, seq_name, bpp=None, metrics=None, prefix_data=None, postfix_list=None):
        out_str = self.get_output_str(seq_name, bpp, metrics, prefix_data, postfix_list)
        self.f.write(f'{out_str}\n')
        self.f.flush()

    def close_file(self):
        if self.f is not None:
            self.f.close()
        self.f = None
