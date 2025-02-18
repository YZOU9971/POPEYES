import numpy as np

class Graph:

    def __init__(self,
                 layout='coco',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.num_node = None
        self.A = None
        self.edge = []
        self.center = None
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        """
        0: Nose
        1: Left Eye
        2: Right Eye
        3: Left Ear
        4: Right Ear
        5: Left Shoulder
        6: Right Shoulder
        7: Left Elbow
        8: Right Elbow
        9: Left Wrist
        10: Right Wrist
        11: Left Hip
        12: Right Hip
        13: Left Knee
        14: Right Knee
        15: Left Ankle
        16: Right Ankle
        (from YOLOv8 official website https://docs.ultralytics.com/tasks/pose/)

        connections:
        0-1 nose to left eye
        0-2 nose to right eye
        1-3 left eye to left ear
        2-4 right eye to right ear
        0-5 nose to left shoulder
        0-6 nose to right shoulder
        5-6 left shoulder to right shoulder
        5-7 left shoulder to left elbow
        6-8 right shoulder to right elbow
        7-9 left elbow to left wrist
        8-10 right elbow to right wrist
        5-11 left shoulder to left hip
        6-12 right shoulder to right hip
        11-12 left hip to right hip
        11-13 left hip to left knee
        12-14 right hip to right knee
        13-15 left knee to left ankle
        14-16 right knee to right ankle

        :param layout:
        :return:
        """
        if layout == 'coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(16, 14), (15, 13), (14, 12), (13, 11), (12, 11), (12, 6), (11, 5),
                             (6, 5), (6, 8), (5, 7), (8, 10), (7, 9), (0, 5), (0, 6), (0, 1), (0, 2),
                             (2, 1), (2, 4), (1, 3)]
            self.edge = self_link + neighbor_link
            self.center = 0
            # TODO: nose being the center may not be the best choice
        else:
            raise ValueError(f'Invalid layout: {layout}')

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))

        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []

            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError(f'Invalid strategy: {strategy}')


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)

    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
