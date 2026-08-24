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
import numpy as np
import torch
from typing import Union, List

class PGXWriter:
    def __init__(self, pgx_file: str, scale_factor: int = 1):
        """
        Initialize the writer.

        Args:
            pgx_file (str): Path to the output PGX file.
            scale_factor (int): Scaling factor for floating-point data (default=1).
        """
        self.pgx_file = pgx_file
        self.scale_factor = scale_factor

    def write(self, data: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Write data to PGX format.

        Args:
            data (torch.Tensor or list of torch.Tensor): Input tensor(s) to write.
        """
        if isinstance(data, torch.Tensor):
            data = [data]

        #os.makedirs(os.path.dirname(self.pgx_file), exist_ok=True)
        channel_output_idx = 0
        with open(self.pgx_file, 'w') as pgx:
            for tensor in data:
                if not torch.is_tensor(tensor):
                    raise ValueError("All inputs must be PyTorch tensors.")

                dtype = tensor.dtype
                bit_depth = self._get_bit_depth(dtype)
                tensor = tensor.detach().cpu().numpy()
                h,w = tensor.shape[-2:]

                endianness = "L" if tensor.dtype.byteorder == '<' or tensor.dtype.byteorder == '=' and np.little_endian else "ML"
                sign = "-" if self._is_signed_type(dtype) else "+"

                for channel_idx in range(tensor.shape[-3]):
                    output_fn = f"{self.pgx_file[:-4]}_{channel_output_idx}"
                    channel_output_idx += 1
                    raw_filename = f"{output_fn}.raw"
                    header_filename = f"{output_fn}.h"
                    cur_data = tensor[..., channel_idx, :, :] 

                    # Scale, round, and clip if float
                    if self._is_float_type(dtype):
                        cur_data = np.clip(
                            np.round(cur_data * self.scale_factor),
                            -(2 ** (bit_depth - 1)),
                            (2 ** (bit_depth - 1)) - 1,
                        ).astype(np.int32 if bit_depth == 32 else np.int64)

                    cur_data.tofile(raw_filename)

                    with open(header_filename, 'w') as header:
                        header.write(f"PG {endianness} {sign}{bit_depth} {w} {h}\n")

                    pgx.write(os.path.basename(raw_filename) + "\n")
            
    def _is_float_type(self, dtype):
        return dtype == torch.float32 or \
               dtype == torch.float64
            
    def _is_signed_type(self, dtype):
        return dtype == torch.int or \
               dtype == torch.int8 or \
               dtype == torch.int16 or \
               dtype == torch.int32 or \
               self._is_float_type(dtype)

    def _get_bit_depth(self, dtype):
        """Return bit depth for the given data type."""
        if dtype == torch.float32:
            return 32
        elif dtype == torch.float64:
            return 64
        elif dtype.is_floating_point:
            raise ValueError(f"Unsupported floating-point type: {dtype}")
        return torch.iinfo(dtype).bits


class PGXReader:
    def __init__(self, pgx_file: str):
        """
        Initialize the reader.

        Args:
            pgx_file (str): Path to the input PGX file.
        """
        self.pgx_file = pgx_file

    def read(self):
        """
        Read tensors from the PGX file.

        Returns:
            list of torch.Tensor: List of tensors read from the PGX file.
        """
        tensors = []
        current_tensors = []
        current_dtype = None
        current_shape = None

        dir_path = os.path.dirname(self.pgx_file)
        with open(self.pgx_file, 'r') as pgx:
            for line in pgx:
                raw_filename = os.path.join(dir_path, line.strip())
                base_name = os.path.splitext(raw_filename)[0]
                header_filename = base_name + ".h"

                with open(header_filename, 'r') as header:
                    header_line = header.readline().strip()
                    _, endianness, sign_bit_depth, width, height = header_line.split(" ")

                    bit_depth = int(sign_bit_depth[1:])
                    signed = sign_bit_depth[0] == "-"
                    dtype = np.dtype(f"{'<' if endianness == 'L' else  '>'}{'i' if signed else 'u'}{bit_depth // 8}")

                # Read raw data
                raw_data = np.fromfile(raw_filename, dtype=dtype).reshape((int(height), int(width)))
                tensor = torch.tensor(raw_data).unsqueeze(0)

                # Group by shape and dtype
                if current_dtype == dtype and current_shape == (int(height), int(width)):
                    current_tensors.append(tensor)
                else:
                    if current_tensors:
                        tensors.append(torch.cat(current_tensors, dim=0))
                    current_tensors = [tensor]
                    current_dtype = dtype
                    current_shape = (int(height), int(width))

        if current_tensors:
            tensors.append(torch.cat(current_tensors, dim=0))

        return tensors

if __name__ == "__main__":

    # Example usage:
    writer = PGXWriter("data.pgx", scale_factor=1000)
    in_tensor = [torch.randn(3, 4, 4).mul(128).to(torch.int32), torch.randn(2, 8, 8).mul(128).to(torch.int32)]
    writer.write(in_tensor)
    print(f"Input tensor: {in_tensor}")

    reader = PGXReader("data.pgx")
    out_tensors = reader.read()
    print(f"Output tensor: {out_tensors}")
    for i, o in zip(in_tensor, out_tensors):
        print( (i == o).all())
