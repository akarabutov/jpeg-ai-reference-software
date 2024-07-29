import argparse
import os

from .metrics import DataClass, MetricsFabric


def main():
    #example usage:
    #python3 -m src.codec.metrics.calc_metric cfp-test-set/00001_TE_1192x832_8bit_sRGB.png rec/VM_00001_TE_1192x832_8bit_sRGB_006.png --bit results/test/bit/VM_00001_TE_006.bits
    ap = argparse.ArgumentParser()
    ap.add_argument('orig', default='', help='Path original YUV/PNG file')
    ap.add_argument('rec', default='', help='Path reconstructed YUV/PNG file')
    ap.add_argument('--bit', default=None, help='Path bitstream file')
    ap.add_argument('--summary', default='summary.txt', help='Path to output suumary file')
    ap.add_argument('--jpegai-psnr',
                    default=False,
                    action='store_true',
                    help='Use 1020 as upper bound for 10 bits')
    ap.add_argument('--internal-bits',
                    type=int,
                    default=8,
                    choices=[8, 10],
                    help='Bits for internal calculations')
    ap.add_argument('--color-conv',
                    default='709',
                    choices=['601', '709', '2020'],
                    help='Color convertion notation')
    ap.add_argument(
        '--metrics_output',
        default=MetricsFabric.metrics_list,
        choices=MetricsFabric.metrics_list,
        nargs='+',
        help=f"Order of metrics in output. Default: [{' '.join(MetricsFabric.metrics_list)}]")
    ap.add_argument('--metrics',
                    default=MetricsFabric.metrics_list,
                    choices=MetricsFabric.metrics_list,
                    nargs='+',
                    help=f"Metrics to be used. Default: [{' '.join(MetricsFabric.metrics_list)}]")
    ap.add_argument('--add',
                    default=[],
                    nargs='+',
                    help='Additional parameters, which will be written to the output file')
    ap.add_argument('--only-print',
                    default=False,
                    action='store_true',
                    help='Only print results to output instead of writing them to the file')
    args = ap.parse_args()

    if args.jpegai_psnr:
        max_vals = {'psnr': 256 * (1 << (args.internal_bits - 8))}
    else:
        max_vals = (1 << args.internal_bits) - 1

    metrics_fab = MetricsFabric(bits=args.internal_bits, max_vals=max_vals)
    metrics = {}
    for m in MetricsFabric.metrics_list:
        metrics[m] = metrics_fab.create_instance(m)

    data_o, _ = DataClass().load_image(args.orig,
                                       def_bits=args.internal_bits,
                                       color_conv=args.color_conv)
    data_r, _ = DataClass().load_image(args.rec,
                                       def_bits=args.internal_bits,
                                       color_conv=args.color_conv)
    ans = [os.path.basename(args.rec)]
    if not os.path.exists(args.bit):
        ans.append('-1')
    else:
        bs_size = os.path.getsize(args.bit)
        bpp = bs_size * 8 / (data_r.shape[-2] * data_r.shape[-1])
        ans.append(str(bpp))

    for m_t in args.metrics_output:
        if m_t in args.metrics:
            m = metrics[m_t]
            tmp = m.calc(data_o, data_r)
            if isinstance(tmp, list):
                ans = ans + [str(x) for x in tmp]
            else:
                ans = ans + [str(tmp)]
        else:
            ans.append('-100')
    ans += args.add

    output_str = '\t'.join(ans)

    if args.only_print:
        print(f'Results: {output_str}\n')
    else:
        with open(args.summary, 'a') as fs:
            fs.write(output_str + '\n')


if __name__ == '__main__':
    main()
