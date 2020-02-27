# faishion_05
python main.py --name faishion_05_ce \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3;
python main.py --name faishion_05_symmetric \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name symmetric --num_classes 3;
python main.py --name faishion_05_bootstrap \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name bootstrap --num_classes 3;
python main.py --name faishion_05_mpe \
--dataset_name FashionMNIST0.5.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 \
--trans_matrix '[[0.5,0.2,0.3],[0.3, 0.5, 0.2],[0.2, 0.3, 0.5]]' --epoch 30 --num_trained_model 1;
python main.py --name faishion_05_mpe_trans \
--dataset_name FashionMNIST0.5.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 --num_trained_model 1;

python main.py --name faishion_05_mpe_trans \
--dataset_name FashionMNIST0.5.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 --num_trained_model 10 \
--pretrained_model_path_format checkpoints/faishion_05_ce_{}/faishion_05_ce_{}_20_model.pth ;

python main.py --name faishion_06_mpe_trans \
--dataset_name FashionMNIST0.6.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 --num_trained_model 10 \
--pretrained_model_path_format checkpoints/faishion_06_ce_{}/faishion_06_ce_{}_20_model.pth ;

python main.py --name faishion_06_ce \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3;
python main.py --name faishion_06_symmetric \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name symmetric --num_classes 3;
python main.py --name faishion_06_bootstrap \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name bootstrap --num_classes 3;
python main.py --name faishion_06_mpe \
--dataset_name FashionMNIST0.6.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3 \
--trans_matrix '[[0.4,0.3,0.3],[0.3, 0.4, 0.3],[0.3, 0.3, 0.4]]';
python main.py --name faishion_06_mpe_trans \
--dataset_name FashionMNIST0.6.npz  --lr 0.001 \
--input_nc 1 --loss_name mpe --num_classes 3;
python main.py --name cifar_ce \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3;
python main.py --name cifar_symmetric \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name symmetric --num_classes 3;
python main.py --name cifar_bootstrap\
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name bootstrap --num_classes 3;
python main.py --name cifar_mpe \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name mpe --num_classes 3 \
--pretrained_model_path checkpoints/cifar_ce/cifar_ce_19_model.pth;
python main.py --name cifar_mpe \
--dataset_name CIFAR.npz  --lr 0.0001 --optimizer adam \
--input_nc 3 --loss_name mpe --num_classes 3 \
--trans_matrix '[[0.4,0.3,0.3],[0.3, 0.4, 0.3],[0.3, 0.3, 0.4]]';



python main.py --name faishion_05_ce_0 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_05_ce_1 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_05_ce_2 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name faishion_05_ce_3 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name faishion_05_ce_4 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name faishion_05_ce_5 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name faishion_05_ce_6 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name faishion_05_ce_7 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name faishion_05_ce_8 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name faishion_05_ce_9 \
--dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";


python main.py --name faishion_06_ce_0 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_1 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_2 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_3 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_4 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_5 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_6 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_7 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_8 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name faishion_06_ce_9 \
--dataset_name FashionMNIST0.6.npz  --lr 0.0003 \
--input_nc 1 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";


python main.py --name cifar_ce_0 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_1 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_2\
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_3 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_4 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_5 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_6 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_7 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_8 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";
python main.py --name cifar_ce_9 \
--dataset_name CIFAR.npz  --lr 0.0003 \
--input_nc 3 --loss_name ce --num_classes 3 --save_model True --save_model True --log_file_path "";

python main.py --name cifar_mpe_trans \
--dataset_name CIFAR.npz  --lr 0.001 \
--input_nc 3 --loss_name mpe --num_classes 3 --num_trained_model 10 \
--pretrained_model_path_format checkpoints/cifar_ce_{}/cifar_ce_{}_20_model.pth ;