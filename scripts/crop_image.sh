data_dir=<folder path of original training images>
lst=<path of image list>

echo mode-I: random cropping
python image_crop.py --lst ${lst} \
                     --data_dir ${data_dir} \
                     --crop_size 1024 \
                     --crop_format random \
                     --save_data_dir <output path of images> \
                     --output_info <output path of image info> \
                     --output_lst <output path of image list> \

#echo mode-II: cropping with sliding window
#python image_crop.py --lst ${lst} \
#                     --data_dir ${data_dir} \
#                     --crop_size 1024 \
#                     --crop_format sliding \
#                     --save_data_dir <output path of images> \
#                     --output_info <output path of image info> \
#                     --output_lst <output path of image list> \
