CUDA_VISIBLE_DEVICES=8 python main.py mode=inference inference.subname=SEMIF_512_very_small inference.input_dir=data/SEMIF/very_small
CUDA_VISIBLE_DEVICES=8 python main.py mode=inference inference.subname=SEMIF_512_small inference.input_dir=data/SEMIF/small
CUDA_VISIBLE_DEVICES=8 python main.py mode=inference inference.subname=SEMIF_512_medium inference.input_dir=data/SEMIF/medium
CUDA_VISIBLE_DEVICES=8 python main.py mode=inference inference.subname=SEMIF_512_medium_large inference.input_dir=data/SEMIF/medium_large
CUDA_VISIBLE_DEVICES=8 python main.py mode=inference inference.subname=SEMIF_512_large inference.input_dir=data/SEMIF/large
CUDA_VISIBLE_DEVICES=8 python main.py mode=inference inference.subname=SEMIF_512_very_large inference.input_dir=data/SEMIF/very_large
CUDA_VISIBLE_DEVICES=8 python main.py mode=inference inference.subname=SEMIF_512_very_very_large inference.input_dir=data/SEMIF/very_very_large
python find_low_greens.py