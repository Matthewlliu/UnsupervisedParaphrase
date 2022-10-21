#CUDA_VISIBLE_DEVICES=3 python augment.py --method corruption
#CUDA_VISIBLE_DEVICES=3 python augment.py --method corruption --augment_size 3 --tag 619

#CUDA_VISIBLE_DEVICES=4 python augment.py --method naive --augment_size 3 --tag 619

CUDA_VISIBLE_DEVICES=6 python augment.py --method paramod --augment_size 3 --tag 618 --datasets SNLI
#CUDA_VISIBLE_DEVICES=7 python augment.py --method paramod --augment_size 10 --tag 616