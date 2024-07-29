python image_crop.py --lst /home/gts/AIcodec/data/jpegai/jpegai_training_set512.txt \
                     --data_dir /home/gts/AIcodec/data/jpegai/jpegai_training/ \
                     --save_data_dir /home/gts/AIcodec/data/jpegai/jpegai_training_random_crop/ \
                     --crop_size 1024 \
                     --crop_format random \
                     --output_info /home/gts/AIcodec/data/jpegai/jpegai_training_set512_random_crop_info.txt \
                     --output_lst /home/gts/AIcodec/data/jpegai/jpegai_training_random_crop/jpegai_training_set512_random_crop.txt \

python image_crop.py --lst /home/gts/AIcodec/data/jpegai/jpegai_training_set512.txt \
                     --data_dir /home/gts/AIcodec/data/jpegai/jpegai_training/ \
                     --save_data_dir /home/gts/AIcodec/data/jpegai/jpegai_training_sliding_crop/ \
                     --crop_size 1024 \
                     --crop_format sliding \
                     --output_info /home/gts/AIcodec/data/jpegai/jpegai_training_set512_sliding_crop_info.txt \
                     --output_lst /home/gts/AIcodec/data/jpegai/jpegai_training_sliding_crop/jpegai_training_set512_sliding_crop.txt \
