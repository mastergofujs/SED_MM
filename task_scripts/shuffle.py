import random

all_files = ['b094', 'b006', 'b093', 'a012', 'a131', 'a128', 'b003', 'b009', 'b091', 'a008', 'b095', 'b005', 'a127',
             'a003', 'b098', 'a124', 'b007', 'a001', 'b099', 'b008', 'b100', 'a013', 'a010', 'a129']

random.shuffle(all_files)
for k in range(4):
    train, val = [], []
    val = all_files[k * 6:(k + 1) * 6]
    for f in all_files:
        if f not in val:
            train.append(f)
    txt_file = open('../aed_data/tut_data/train_val_split/f' + str(k + 1) + '.txt', 'w')
    txt_file.write('train:'+','.join(train)+'\n')
    txt_file.write('val:'+','.join(val))
