import subprocess

def get_unused_gpu():
    stdout = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8').split('\n')
    all_dev, dev_in_use, flag = set(), set(), False
    for idx, line in enumerate(stdout):
        if line.startswith('| N/A'):
            all_dev.add(int(stdout[idx-1][1:7]))
            continue
        if line.startswith('|====') and '+' not in line:
            flag = True
            continue
        if flag and line.strip() and not line.startswith('+-------'):
            dev_in_use.add(int(line[1:10]))
            continue
    return sorted(list(all_dev.difference(dev_in_use)))

if __name__ == '__main__':
    print(get_unused_gpu())