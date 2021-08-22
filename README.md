## Requirements
```
torch==1.2.0
tensorboadX==1.6
```

## Run
To reproduce the results on FB15k, FB15k-237 and NELL, the hyperparameters are set in `example.sh`.
```
bash example.sh
```

Note that GQE is equal to transe in the code


Before running the code, go to codes/trans_matrix_gen/ and run gen_trans_matrix_multi_from_kg_triple.py, gen_trans_matrix_single_kg_triple.py.
Then, move the generated files to the corresponding data dir.
For example, if the file_path in gen_trans_matrix_single_kg_triple.py and gen_trans_matrix_multi_from_kg_triple.py is ../../data/FB15k, then moving the generated file to data/FB15k



Parameter when running the model in PyCharm Directly:

For NewLook

--cuda --do_train --do_valid --do_test --do_query --data_path ../data/NELL --model BoxNewLook -n 128 -b 512 -d 400 -g 24 -a 1.0 -lr 0.0001 --max_steps 50000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc --stepsforpath 50000  --offset_deepsets inductive --center_deepsets eleattention --print_on_screen

For Q2B

--cuda --do_train --do_valid --do_test --do_query --data_path ../data/NELL --model BoxTransE -n 128 -b 512 -d 400 -g 24 -a 1.0 -lr 0.0001 --max_steps 50000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc --stepsforpath 50000  --offset_deepsets inductive --center_deepsets eleattention --print_on_screen

For GQE

--cuda --do_train --do_valid --do_test --do_query --data_path ../data/NELL --model TransE -n 128 -b 512 -d 400 -g 24 -a 1.0 -lr 0.0001 --max_steps 50000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 --geo vec --task 1c.2c.3c.2i.3i.ic.ci.2u.uc.2d.3d.dc --stepsforpath 50000  --offset_deepsets inductive --center_deepsets eleattention --print_on_screen

