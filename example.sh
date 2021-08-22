#!/bin/bash

 CUDA_VISIBLE_DEVICES=0 python -u codes/run_model_NewLook.py --do_train --cuda --do_valid --do_test \
  --data_path data/FB15k --model BoxNewLook -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 50000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 \
  --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc --stepsforpath 50000  --offset_deepsets inductive --center_deepsets eleattention \
  --print_on_screen

 CUDA_VISIBLE_DEVICES=1 python -u codes/run_model_NewLook.py --do_train --cuda --do_valid --do_test \
  --data_path data/FB15k-237 --model BoxNewLook -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 50000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 \
  --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc --stepsforpath 50000  --offset_deepsets inductive --center_deepsets eleattention \
  --print_on_screen

 CUDA_VISIBLE_DEVICES=2 python -u codes/run_model_NewLook.py --do_train --cuda --do_valid --do_test \
  --data_path data/NELL --model BoxNewLook -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 50000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 \
  --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc --stepsforpath 50000  --offset_deepsets inductive --center_deepsets eleattention \
  --print_on_screen
