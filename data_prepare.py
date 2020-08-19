from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import os

def load_mnist(data_dir):
    dataset = input_data.read_data_sets(data_dir, one_hot=True)

    def split_task(dt):
        task_dict = {i:[] for i in range(10)}
        for i in range(dt.labels.shape[0]):
            x = dt.images[i, :]
            y = dt.labels[i, :]
            lb = int(np.argmax(y))
            if lb not in [0, 1, 2, 4]:
                continue
            x = np.reshape(x, [28, 28])[:27, :27]
            x = x - np.mean(x) * np.ones_like(x)
            dim10_feat = np.zeros([9, 9, 10])
            for ii in range(9):
                for jj in range(9):
                    patch_ij = np.reshape(x[ii*3:(ii+1)*3, jj*3:(jj+1)*3], [9])
                    if float(np.sum(patch_ij)) == 0.0:
                        normalized_ij = np.concatenate([np.array([1.0]), patch_ij])
                    else:
                        normalized_ij = np.concatenate([np.array([0]), patch_ij / np.sqrt(np.sum(np.square(patch_ij)))])
                    dim10_feat[ii, jj] = normalized_ij
            task_dict[lb].append(np.expand_dims(dim10_feat, axis=0))

        dataset_dict = {}
        for i in [0,1,2,4]:
            dataset_dict[i] = np.concatenate(task_dict[i], axis=0)
        return dataset_dict

    train_dict = split_task(dataset.train)
    test_dict = split_task(dataset.test)
    return train_dict, test_dict

def get_train_test():
    if not os.path.exists("./feature_input.pkl"):
        train_dict, test_dict = load_mnist("./data")
        pickle.dump([train_dict, test_dict], open("./feature_input.pkl", mode="wb"))
    else:
        train_dict, test_dict = pickle.load(open("feature_input.pkl", mode="rb"))
    return train_dict, test_dict