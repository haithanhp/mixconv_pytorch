python train_cifar.py  --lr 0.1 --batch-size 256 -a mixnet-s
python train_cifar.py --lr 0.016 --batch-size 256 -a mixnet-s --dtype cifar100 --optim adam --scheduler exp --epochs 650