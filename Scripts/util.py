import torch

def count_line(file_name):
    with open(file_name) as f:
        for count, _ in enumerate(f, 1):
            pass
    return count


def get_cluster_indices_list(clusters):
    cluster_indices_list = []
    for cluster in clusters:
        cluster_index_list = []
        for indices in cluster:
            if indices[0] == indices[1]:
                cluster_index_list.append(indices[0])
            else:
                for n in range(indices[1] - indices[0] + 1):
                    cluster_index_list.append(n + indices[0])
        cluster_indices_list.append(cluster_index_list)
    return sum(cluster_indices_list, [])


def extract_file_name(filepath):
    import re
    return(re.findall(r'(?<=\/)(?:(?:\w+\.)*\w+)$', filepath))

'''def get_file_from_google_cloud_storage(filepath):
    import subprocess
    import os
    tmp_file = os.path.join('/tmp', extract_file_name(filepath)[0])
    subprocess.check_call(['gsutil', 'cp', filepath, tmp_file])
    return tmp_file'''


# load checkpoint to resume training
def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['val_loss'].item()


