import numpy as np


def convert_transaction_to_seq(conn, item='condition_concept_id', saveFileName=None):
    from CDM_DataSetMaker_ops import query, dumpingFiles        
    
    q = str("select person_id, rn, {} " 
            + " from LJ_HF_replica_study.dbo.treat_seq order by person_id, rn")
    results = query(conn, q.format(item))
    
    p_seq = []
    v_seq = []
    d_seq = []
    prev_pid = results[0][0]
    prev_rn = 1
    for row in results:        
        if prev_pid!=row[0]:
            v_seq.append(d_seq)
            p_seq.append(v_seq)
            v_seq = []
            d_seq = []
            prev_rn = 1

        if prev_rn==row[1]:
            values = row[2:]
            if ('condition_concept_id'==item):
                values = values[0]
                d_seq.append(values)
            else:
                if (row[1] == 1) & (len(d_seq)==0):
                    d_seq.append(values)
        else:
            v_seq.append(d_seq)
            values = row[2:]
            if ('condition_concept_id'==item):
                values = values[0]
            d_seq = [values]
        prev_rn = row[1]
        prev_pid = row[0]
    v_seq.append(d_seq)
    p_seq.append(v_seq)
    
    if saveFileName is not None:
        dumpingFiles('./', saveFileName, p_seq)
    return p_seq

def truncated_data(dx_seqs, cut_num):
    truncated_d_seq = []
    for d_seq in dx_seqs:
        if cut_num<=len(d_seq):
            truncated_d_seq.append(d_seq)
        else:
            print(d_seq)
    return truncated_d_seq

def code_to_id(data):
    from itertools import chain
    code_to_id = {code: i for i, code in enumerate(set(chain.from_iterable(chain.from_iterable(data))))}
    print('code_size: ', len(code_to_id))
    return code_to_id

class med2vec_DataSet():
    def __init__(self, d_seqs, code_to_id, left_window_size, right_window_size):
        self._num_examples = len(d_seqs)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.d_seqs = d_seqs
        self.code_to_id = code_to_id
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size
        self.multihot_seqs = self._convert_to_sparseMultiHot(d_seqs)
        
    def _convert_to_sparseMultiHot(self, data):
        from scipy.sparse import csr_matrix
        import numpy as np

        pid_multihot = []
        for pid, dxs in enumerate(data):
            pid_row = []
            pid_col = []
            for i, codes in enumerate(dxs):
                for c in codes:
                    pid_row.append(i)
                    pid_col.append(self.code_to_id[c])
            visit_times=len(dxs)
            pid_multihot.append([csr_matrix((np.ones(len(pid_col), np.float32), (pid_row, pid_col)), 
                                           shape=(visit_times, len(self.code_to_id))), visit_times])
        return pid_multihot

    def _shuffle(self, d_seqs):
        import sklearn as sk
        return sk.utils.shuffle(self.d_seqs)
    
    def _edge_extractor(self, sparse_list):
        pid_input_label = []
        seq_list = sparse_list.toarray()
        max_visit_times = len(seq_list)
        for i, seq in enumerate(seq_list):
            inputs = seq
            left_labels = seq_list[max(0, i-self.left_window_size):i] 
            right_labels = seq_list[i+1:min(max_visit_times, i+self.right_window_size)+1]
            labels = np.concatenate((left_labels,right_labels), axis=0)
            for label in labels:
                pid_input_label.append([inputs, label])
        return np.array(pid_input_label)
    
    def _cooccur_idx(self, sparse_matrices):
        from itertools import permutations, chain
        import numpy as np

        permu_per_row = []
        for i in range(len(sparse_matrices[0].indptr)-1):
            col_idx_per_row = sparse_matrices[0].indices[sparse_matrices[0].indptr[i]:sparse_matrices[0].indptr[i+1]]
            result = list(permutations(col_idx_per_row, 2))
            if len(result):
                permu_per_row.append(result)
        permu = np.array(list(chain.from_iterable(permu_per_row)))
        if len(permu):
            return permu[:,0], permu[:,1]
        else:
            return [], []
    
    def next_batch(self):
        if self._index_in_epoch>(self._num_examples-1):
            self._index_in_epoch = 0
            #self.multihot_seqs = self._shuffle(self.multihot_seqs)
            #print(self.multihot_seqs)
        batch_mulitihot, visit_times = self.multihot_seqs[self._index_in_epoch]
        batch_cooccur_idx_i, batch_cooccur_idx_j = self._cooccur_idx(self.multihot_seqs[self._index_in_epoch])
        batch_inputs, batch_labels = self._edge_extractor(batch_mulitihot)[:,0], self._edge_extractor(batch_mulitihot)[:,1]
        self._index_in_epoch += 1
        return batch_inputs, batch_labels, visit_times, batch_cooccur_idx_i, batch_cooccur_idx_j



