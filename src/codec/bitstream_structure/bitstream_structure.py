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

##
import hashlib
from io import BytesIO
from typing import List, Tuple
import numpy as np
from collections import OrderedDict
from src.codec.entropy_coding.lib_wrappers.ec_lib_base import ECLibBase, ECLibBaseWithThread
from src.codec.entropy_coding import ECLibDirect, ECLibEmpty, StreamBitWriter, StreamBitReader, Binarizers

from src.codec.common import TensorOps
from .substream import Substream
from .layouts_def import SubstreamLayouts
from .aemem import AEMemObject


        
    
   
class BitstreamStructure:
    
    BYTE_ORDER = 'big'
    
    def __init__(self, ae_factory: "ECComposite", *ae_args, **ae_kwargs):
        self.ae_mems_dict = OrderedDict()
        self.verbose = getattr(ae_factory, "verbose", False)
        self.dump_tool = ae_kwargs.get('dump_tool', None)
        self.reverse_encode_order = getattr(ae_factory, 'reverse_encode_order', False)
        self.ae_factory = ae_factory
        self.ae_args = ae_args
        self.ae_kwargs = ae_kwargs        
        self.coder_direction = ae_kwargs.get('coder_direction', None)   # Coder status: 0 is encoder, 1 is decoder
        self.substreams = list()
        self.threads_count_names = {
            SubstreamLayouts.MARKER_SOZ: "num_threads_z",
            SubstreamLayouts.MARKER_SOQ: "num_threads_q", 
            SubstreamLayouts.MARKER_SORP: "num_threads_r_0",
            SubstreamLayouts.MARKER_SORS: "num_threads_r_1"
        }        
        
        
    @staticmethod
    def get_hash(data: bytes) -> str:
        md5 = hashlib.md5()
        md5.update(data)
        return md5.hexdigest()
        
        
    def read_substreams(self, input_fn: str):
        """Read bitstream file to a set of substreams

        Args:
            input_fn (str): name of the input bitstream file
        """
        with open(input_fn, "rb") as f:
            EOC_found = False
            SOC_found = False
            while (not EOC_found):
                new_subs = Substream(self.BYTE_ORDER)
                read_ok = new_subs.read(f)
                if not SOC_found and not read_ok:
                    SOC_found = (new_subs.marker_id == SubstreamLayouts.MARKER_SOC)
                elif not read_ok:
                    EOC_found = (new_subs.marker_id == SubstreamLayouts.MARKER_EOC)
                else:
                    if len(self.substreams) == 0:
                        # Picture header should be the first substream after SOC
                        assert new_subs.marker_id == SubstreamLayouts.MARKER_PIH
                    self.substreams.append(new_subs)
                    
        if self.verbose:
            for s in self.substreams:
                substream_type = SubstreamLayouts.get_substreamtype_by_markerid(s.marker_id)
                print(f"Read substream {substream_type.human_readable_name} with marker {hex(substream_type.marker_id)} with size {len(s.mem_stream.getvalue())} bytes and hash {self.get_hash(s.mem_stream.getvalue())}")
        

    def write_substreams(self, output_fn: str):
        """Write all substreams to the output file

        Args:
            output_fn (str): name of the output bitstream file
        """
        with open(output_fn, "wb") as f:
            Substream.encode_marker_id(f, SubstreamLayouts.MARKER_SOC)
            for i, s in enumerate(self.substreams):
                if i == 0:
                    assert s.marker_id == SubstreamLayouts.MARKER_PIH
                s.write(f)
                if self.verbose:
                    substream_type = SubstreamLayouts.get_substreamtype_by_markerid(s.marker_id)
                    print(f"Store substream {substream_type.human_readable_name} with marker {hex(substream_type.marker_id)} with size {len(s.mem_stream.getvalue())} bytes and hash {self.get_hash(s.mem_stream.getvalue())}")        
                
            Substream.encode_marker_id(f, SubstreamLayouts.MARKER_EOC)
            
    def substreams_total_size(self) -> int:
        ans = 0
        for s in self.substreams:
            ans += s.get_substream_size()
        return ans
            
    def _create_aemem(self, substream_type: SubstreamLayouts.SubstreamType, max_mem_size: int = None) -> AEMemObject:
        marker_id = substream_type.marker_id
        mid_idx = SubstreamLayouts.get_substream_idx(marker_id)
        mem_max_sizes = getattr(self.ae_factory, "max_compressed_size", [None]*SubstreamLayouts.get_substream_type_count())
        if max_mem_size is None:
            max_mem_size = mem_max_sizes[mid_idx]
        additional_params = dict()
        num_threads = 1
        if marker_id in self.threads_count_names:
            num_threads = getattr(self.ae_factory, self.threads_count_names[marker_id], 1)
            additional_params['num_threads'] = num_threads
        ae = self.ae_factory(*self.ae_args, verbose=self.verbose, **self.ae_kwargs, **additional_params) if substream_type.use_ae else ECLibDirect(verbose=self.verbose, substream_name=substream_type.human_readable_name)
        return AEMemObject(ae, max_mem_size, num_threads, verbose=self.verbose)
    
            
    def get_ec(self, substream_type: SubstreamLayouts.SubstreamType, region_idx: int = 0) -> ECLibBase:
        marker_id = substream_type.marker_id
        ans = None
        self._fill_absent_regions(marker_id, region_idx)
        cur_ae_mem_lst = self.ae_mems_dict.get(marker_id, list())
        if cur_ae_mem_lst[region_idx] is None:
            if self.coder_direction == 0:
                ans = self._create_aemem(substream_type)
                ans.encode_init()
            else:
                ae = ECLibEmpty()
                ae.decode_init()
                ans = AEMemObject(ae, 0, 1)
            cur_ae_mem_lst[region_idx] = ans
        else:
            ans = cur_ae_mem_lst[region_idx]
        return ans.ae
    
    def _fill_absent_regions(self, marker_id: int, region_id: int = 0) -> None:
        cur_ae_mem_lst = self.ae_mems_dict.get(marker_id, list())
        diff = (region_id+1) - len(cur_ae_mem_lst)
        if diff > 0:
            #assert (self.coder_direction is not None) and (self.coder_direction == 0)
            for _ in range(diff):
                cur_ae_mem_lst.append(None)
            self.ae_mems_dict[marker_id] = cur_ae_mem_lst

    def _set_aemem(self, obj: AEMemObject, marker_id: int, region_id: int = 0):
        self._fill_absent_regions(marker_id, region_id)
        cur_ae_mem_lst = self.ae_mems_dict.get(marker_id, list())
        cur_ae_mem_lst[region_id] = obj        
    
    
    def parse_substreams(self, only_non_ae: bool = None) -> None:
        """Parse substreams and store data to mem

        Args:
            only_non_ae (bool, optional): parse only non AE substreams. In a case of None it parses all substreams. Defaults to None.
        """
        num_regions = getattr(self.ae_factory, "num_regions", 1)
        region_residual_in_its_own_substream_flag = getattr(self.ae_factory, "region_residual_in_its_own_substream_flag", 0)        
        
        self.coder_direction = 2
        for s in self.substreams:
            region_idx = None
            marker_id = s.marker_id           
            layout_type = SubstreamLayouts.get_substreamtype_by_markerid(marker_id)
            if only_non_ae is not None and (only_non_ae == layout_type.use_ae):
                continue
            
            mem_sizes = list()
            s.mem_stream.seek(0)
            region_idx = 0
            if ( (only_non_ae is None) or not only_non_ae) and layout_type.has_regions:
                if region_residual_in_its_own_substream_flag == 1:
                    region_idx = int.from_bytes(s.mem_stream.read(1), s.BYTE_ORDER)
                    mem_sizes.append( (region_idx, s.get_substream_size()-1) ) 
                else:
                    bypass_coder = StreamBitReader(s.mem_stream, s.BYTE_ORDER)
                    total_size = 0
                    for i in range(num_regions-1):
                        val = Binarizers.decode_unsigned_expgolomb_k0(bypass_coder)
                        mem_sizes.append( (i, val) )
                        total_size += val
                    mem_sizes.append( (num_regions-1, s.get_substream_size() - total_size - s.mem_stream.tell()) )
                    if self.dump_tool is not None:
                        region_threads_lst = list()
                        for _ in range(len(mem_sizes)):
                            region_threads_lst.append(None)
                        for (id,mem_s) in mem_sizes:
                            region_threads_lst[id] = mem_s
                        params = OrderedDict()
                        params['Subregions'] = region_threads_lst
                        self.dump_tool.add_substream_params(layout_type.human_readable_name, region_idx, params)
                        
            else:
                mem_sizes.append( (0, s.get_substream_size()) ) 
            if self.dump_tool is not None:
                params = OrderedDict()
                params['Size'] = s.get_substream_size()
                self.dump_tool.add_substream_params(layout_type.human_readable_name, region_idx, params)
                
            for (reg_id, mem_size) in mem_sizes:
                obj = self._create_aemem(layout_type, mem_size)
                obj.parse_substream(BytesIO(s.mem_stream.read(mem_size)), s.BYTE_ORDER)
                self._set_aemem(obj, marker_id, reg_id)
        
    def decode_term(self):
        for aemem_lst in self.ae_mems_dict.values():
            for aemem in aemem_lst:
                if aemem is not None:
                    aemem.decode_term()
                
    def fill_substreams(self) -> None:
        """Move data from memory to substreams
        """
        region_residual_in_its_own_substream_flag = getattr(self.ae_factory, "region_residual_in_its_own_substream_flag", 0)
        self.substreams = list()
        for marker_id, ae_mems_lst in self.ae_mems_dict.items():
            substream_type = SubstreamLayouts.get_substreamtype_by_markerid(marker_id)
            # Finalize all encoders and store data to separate memories
            regions_mem = list()
            for region_idx, cur_aemem in enumerate(ae_mems_lst):
                process_mem = True
                if region_residual_in_its_own_substream_flag == 0 and substream_type.has_regions:
                    assert cur_aemem is not None
                else:
                    process_mem = cur_aemem is not None
                    
                if process_mem:
                    region_mem = None
                    if cur_aemem.is_used():
                        region_mem = BytesIO()
                        if region_residual_in_its_own_substream_flag == 1 and substream_type.has_regions:
                            region_mem.write(region_idx.to_bytes(1, self.BYTE_ORDER))
                        cur_aemem.store_substream(region_mem, self.BYTE_ORDER)
                    if region_mem is not None:
                        regions_mem.append(region_mem)

            new_subs_mem = None           
            if region_residual_in_its_own_substream_flag == 0 and substream_type.has_regions:
                new_subs_mem = BytesIO()
                # Store "offsets" (sizes) of the regions
                bypass_coder = StreamBitWriter(new_subs_mem, self.BYTE_ORDER)
                for region_mem in regions_mem[:-1]:
                    Binarizers.encode_unsigned_expgolomb_k0(bypass_coder, len(region_mem.getvalue()))
                bypass_coder.flush()
                
            # Store data of offsets
            for region_mem in regions_mem:
                if new_subs_mem is None:
                    self.substreams.append(Substream(self.BYTE_ORDER, marker_id, region_mem))
                else:
                    new_subs_mem.write(region_mem.getvalue())
                    
            if new_subs_mem is not None:
                self.substreams.append(Substream(self.BYTE_ORDER, marker_id, new_subs_mem))