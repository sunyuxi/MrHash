1. train a myautoencder

    cd myautoencoder; python autoencoder.py 

2. train MrHash

    python train.py --hash_bit 32 --pretrained_dp_path your_trained_ae_path

3. test MrHash

    python test.py --hash_bit --pretrain_path your_trained_mrhash
