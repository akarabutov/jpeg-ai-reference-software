from argparse import ArgumentParser
from src.codec.common import Image

def get_args():
    parser = ArgumentParser()
    parser.add_argument('input_file', type=str, help="Path to input file")
    parser.add_argument('output_file', type=str, help="Path to output file")
    parser.add_argument('bitdepth', type=int, default=8, help="Bit-depth of input file")
    parser.add_argument('--fill-bit', type=int, default=0, choices=[0,1], help="Bit for filling LSB")
    return parser.parse_args()
    
    
def convert_bits(input_fn, output_fn, outptu_bits, fill_bit=0):
    img = Image.read_file(input_fn)    
    img.write_file(output_fn, outptu_bits, fill_bit=fill_bit)
    
def main():
    args = get_args()
    convert_bits(args.input_file, args.output_file, args.bitdepth, args.fill_bit)
    
if __name__ == "__main__":
    main()