
path1 = '/user/home/wy20260/scratch/individualProject/darknet_authorship_verification_agora_anon/darknet_authorship_verification_val_nodupe_anon.jsonl'

path2='/user/home/wy20260/scratch/individualProject/darknet_authorship_verification_silkroad1_anon/darknet_authorship_verification_val_nodupe_anon.jsonl'

path3 = '/user/home/wy20260/scratch/individualProject/darkreddit_authorship_verification_anon/darkreddit_authorship_verification_val_nodupe_anon.jsonl'

path4 = '/user/home/wy20260/scratch/individualProject/all/merged_val.jsonl'


with open(path1, 'r') as f1, \
     open(path2, 'r') as f2, \
     open(path3, 'r') as f3, \
     open(path4, 'w') as f4:

    # Iterate over each line in file1 and write it to merged file
    for line in f1:
        f4.write(line)
    print(1)
    # Iterate over each line in file2 and write it to merged file
    for line in f2:
        f4.write(line)
    print(2)
    # Iterate over each line in file2 and write it to merged file
    for line in f3:
        f4.write(line)
    print(3)


