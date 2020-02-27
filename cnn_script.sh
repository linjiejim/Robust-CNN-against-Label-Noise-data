# GPU Environment
CUDA_VISIBLE_DEVICES=0

# faishion_05
python main.py --name faishion_05_ce \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3

python main.py --name faishion_05_symmetric \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name symmetric --num_classes 3

python main.py --name faishion_05_bootstrap \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name bootstrap --num_classes 3

python main.py --name faishion_05_mpe \
--dataset_name FashionMNIST0.5.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 \
--trans_matrix '[[0.5,0.2,0.3],[0.3, 0.5, 0.2],[0.2, 0.3, 0.5]]'

python main.py --name faishion_05_mpe \
--dataset_name FashionMNIST0.5.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 \
--pretrained_model_path checkpoints/faishion_05_ce/faishion_05_ce_19_model.pth

# faishion_06
python main.py --name faishion_06_ce \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3

python main.py --name faishion_06_symmetric \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name symmetric --num_classes 3

python main.py --name faishion_06_bootstrap \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name bootstrap --num_classes 3

python main.py --name faishion_06_mpe \
--dataset_name FashionMNIST0.6.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 \
--trans_matrix '[[0.4,0.3,0.3],[0.3, 0.4, 0.3],[0.3, 0.3, 0.4]]'

python main.py --name faishion_06_mpe \
--dataset_name FashionMNIST0.6.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 \
--pretrained_model_path checkpoints/faishion_06_ce/faishion_06_ce_19_model.pth

# faishion_ce
python main.py --name cifar_ce \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3

python main.py --name cifar_symmetric \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name symmetric --num_classes 3

python main.py --name cifar_bootstrap\
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name bootstrap --num_classes 3

python main.py --name cifar_mpe \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name mpe --num_classes 3 \
--pretrained_model_path checkpoints/cifar_ce/cifar_ce_19_model.pth

python main.py --name cifar_mpe \
--dataset_name CIFAR.npz  --lr 0.0001 --optimizer adam \
--input_nc 3 --loss_name mpe --num_classes 3 \
--trans_matrix '[[0.4,0.3,0.3],[0.3, 0.4, 0.3],[0.3, 0.3, 0.4]]'

# test example
python main.py --name faishion_06_ce \
--dataset_name FashionMNIST0.6.npz --is_testing \
--input_nc 1 --loss_name ce --num_classes 3 \
--pretrained_model_path checkpoints/faishion_06_ce/faishion_06_ce_19_model.pth

python main.py --name faishion_05_ce \
--dataset_name FashionMNIST0.5.npz --is_testing \
--input_nc 1 --loss_name ce --num_classes 3 \
--pretrained_model_path checkpoints/faishion_05_ce/faishion_05_ce_19_model.pth