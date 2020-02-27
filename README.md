# COMP5328Assignment2

 
Use shell script to run the code.
For training a model, use script such as:

```
python main.py --name faishion_05_symmetric \
    --dataset_name FashionMNIST0.5.npz  --lr 0.0003 \
    --input_nc 1 --loss_name ce --num_classes 3;
```

For testing a model, use script such as:
```
python main.py --name faishion_06_symmetric \
    --dataset_name FashionMNIST0.6.npz --is_testing \
    --input_nc 1 --loss_name ce --num_classes 3 \
    --pretrained_model_path checkpoints/faishion_06_symmetric/faishion_06_symmetric_20_model.pth
```

The meanings of availble parameters are:
1. `--name` name of the experiment. It decides where to store samples and models
2. `--dataset_name` the name of the dataset
3. `--dataset_root` the root folder of the dataset
4. `--checkpoints_dir` the root folder of checkpoints
5. `--input_nc` # of input image channels
6. `--nf` # of gen filters in first conv layer
7. `--num_classes` # of classes
8. `--is_testing` if it is testing phase
9. `--epoch` training epochs
10. `--batch_size` training batch size
11. `--loss_name` name of the loss function used to train the model bootstrap|ce|mpe|symmetric.
12. `--sigma` sigma value in the loss function.
13. `--lr` initial learning rate for adam
14. `--beta1` momentum term of adam
15. `--beta2` momentum term of adam
16. `--val_split_rate` the split rate of data for validating
17. `--optimizer` the optimizer method sgd|adam
18. `--print_loss` If loss should be printed during the training phase
19. `--save_model` If model need to be saved after training
20. `--num_trained_model` How many model do we need to train
21. `--trans_matrix` transition matrix if known
22. `--pretrained_model_path` the path of the pretrained model 
23. `--pretrained_model_path_format` the path format of the pretrained model 
24. `--log_file_path` the path of the log file

