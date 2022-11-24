1. Build a myautoencder
```
cd myautoencoder; python autoencoder.py
```

2. Train MrHash
```
python train.py --hash_bit 32 --pretrained_dp_path your_trained_ae_path
```

3. Test MrHash
```
python test.py --hash_bit 32 --pretrain_path your_trained_mrhash
```
