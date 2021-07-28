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
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, counter=0, delta=0, path='BestModel', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        import numpy as np
        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, mode):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, mode)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, mode)
            self.counter = 0
        torch.save(self.counter, "{}/{}_EarlyStopping_counter.pt".format(self.path,mode))

    def save_checkpoint(self, val_loss, model, mode):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(model.state_dict(), "{}/{}.pt".format(self.path, mode))
        self.val_loss_min = val_loss


