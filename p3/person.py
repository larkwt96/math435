import numpy as np


class Person:
    # don't include the 1 or 0. 2 classes will have 1 partition marker. the
    # partitions don't sum to 1, they accumulate:
    # [0.1, 0.2, 0.3, 0.4, 0.5, ..., 0.9] for example.
    class_partition = [0.5]
    num_classes = len(class_partition) + 1
    class_names = ['Democrats', 'Republicans']

    def __init__(self):
        # assume class_ind is 0 indexed
        self.class_ind = self.get_class()

    @staticmethod
    def get_class_name(class_ind):
        return Person.class_names[class_ind]

    @staticmethod
    def get_class():
        rnd = np.random.rand()
        for sel, e in enumerate(Person.class_partition + [1]):
            if rnd < e:
                return sel
        raise ValueError('class_partition must be a partition')
