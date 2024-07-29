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


class SubstreamLayouts:
    """
    Class with description of substreams
    """

    """
    A list of markers of substreams
    """
    MARKER_SOC = 0xFF80
    MARKER_EOC = 0xFF81
    MARKER_PIH = 0xFF82
    MARKER_TON = 0xFF83
    MARKER_RDI = 0xFF84
    MARKER_SOZ = 0xFF88
    MARKER_SORP= 0xFF89
    MARKER_SORS= 0xFF8A
    MARKER_SOQ = 0xFF8B
    MARKER_UDI = 0xFF8C
    
    class SubstreamType:
        """
        Class for storing information about substream
        """
        
        def __init__(self, marker_id: int, use_ae: bool, use_threads: bool, mandatory_substream: bool, has_regions: bool, human_readable_name: str):
            self.marker_id = marker_id
            self.use_ae = use_ae
            self.use_threads = use_threads
            self.mandatory_substream = mandatory_substream
            self.has_regions = has_regions
            self.human_readable_name = human_readable_name
        
    
    """
    Actual substreams of the bitstream
    """
    FullmarkersDict = [
        # Fields:     MARKER_ID     USE_AE      USE_THREADS (for AE)        MANDATORY_SUBSTREAM      HAS REGIONS        SUBSTREAM_NAME
        SubstreamType(MARKER_PIH,   False,      False,                      True,                    False,             "picture_header"),             # Picture header
        SubstreamType(MARKER_TON,   False,      False,                      False,                   False,             "tool_header"),                # Tools header
        SubstreamType(MARKER_SOQ,   True,       True,                       False,                   False,             "quality_map"),                # Quality map
        SubstreamType(MARKER_SOZ,   True,       True,                       True,                    False,             "z_substream"),                # Z substream
        SubstreamType(MARKER_SORP,  True,       True,                       True,                    True,              "r_prim_substream"),           # R substream (Primary)
        SubstreamType(MARKER_SORS,  True,       True,                       True,                    True,              "r_sec_substream"),            # R substream (Secondary)
        SubstreamType(MARKER_UDI,   False,      False,                      False,                   False,             "udi"),                        # User defined information
        SubstreamType(MARKER_RDI,   False,      False,                      False,                   False,             "rendering_information"),      # Rendering information
    ]
    
    """
    Links between substream's name in SW interface (for the primary component) to substream in the bitstream
        "NAME OF STREAM PART":      index in FullmarkersDict
    """
    MarkersPrimary = {
        "pic_header":   0,
        "tool_header":  1,
        "qmap":         2,
        "z":            3,
        "r":            4,
        "udi":          6,
        "rdi":          7,
    }

    """
    Links between substream's name in SW interface (for the secondary component) to substream in the bitstream
        "NAME OF STREAM PART":      index in FullmarkersDict
    """
    MarkersSecondary = {
        "pic_header":   0,
        "tool_header":  1,
        "qmap":         2,
        "z":            3,
        "r":            5,
        "udi":          6,
        "rdi":          7,
    }
    
    @staticmethod
    def get_substream_type_count() -> int:
        return len(SubstreamLayouts.FullmarkersDict)
        
    @staticmethod
    def get_substreamtype_by_name(is_primary: bool, substream_type: str) -> SubstreamType:
        marker = SubstreamLayouts.MarkersPrimary if is_primary else SubstreamLayouts.MarkersSecondary
        substream_idx = marker.get(substream_type)
        return SubstreamLayouts.FullmarkersDict[substream_idx]

    @staticmethod
    def get_substreamtype_by_markerid(marker_id: int) -> SubstreamType:
        for s in SubstreamLayouts.get_substream_type_gen():
            if s.marker_id == marker_id:
                return s
        return None

           
    @staticmethod
    def get_substream_type_gen():
        """Go over all possible markers
        """
        for i in SubstreamLayouts.FullmarkersDict:
            yield i
            
    @staticmethod
    def get_substream_by_id(stream_id: int) -> SubstreamType:
        """Get substream configuration by its id (see index in FullmarkersDict)
        stream_id (int): index in FullmarkersDict
        """
        return SubstreamLayouts.FullmarkersDict[stream_id]
    
    @staticmethod
    def get_substream_idx(marker_id: int) -> int:
        """Get substream index in FullmarkersDict by substream marker
        """
        for i, s in enumerate(SubstreamLayouts.get_substream_type_gen()):
            if s.marker_id == marker_id:
                return i
        return None
    
    
    