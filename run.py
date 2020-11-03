import subprocess
import os


current_path = os.getcwd()
path = os.getcwd() + "/csvs/default"
print(path)
all_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        all_files.append(os.path.join(root,file))


ran_test = []

for f in all_files:
    if "csv" in f and "_pop_" in f and "_k_" in f and "_n_" in f and "seed" in f and "iteration" in f:
      f = f[f.rfind('/'):]
      print(f)
      iteration = int(f[f.find("iteration")+10:f.find("_pop")])
      pop       = int(f[f.find("_pop_")+5:f.find("_k_")])
      k         = int(f[f.find("_k_")+3:f.find("_n")])
      n         = int(f[f.find("_n_")+3:f.find("_seed")])
      seed      = int(f[f.find("_seed_")+ 6 :f.find("_single")])
      ran_test.append((iteration,pop,k,n,seed))


n_list = [1,10,50,100,150,200,250,300,350]
k_list = [1,10,50,100,150,200,250,300,350]
iterations = [100,200]
pop_size = 300
seed = 1234
os.chdir(current_path)
for i in iterations:
    for k in k_list[::-1]:
        for n in n_list[::-1]:
            new_test = (i,pop_size, k, n, seed)     
            command = "python main.py --output-csv --num-gen {} -k {} -n {} -p {} ".format(i, k,n, pop_size)
            print(command)
            if not new_test in ran_test:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out , err = process.communicate()
                errcode = process.returncode
            print("\t compleated")             
