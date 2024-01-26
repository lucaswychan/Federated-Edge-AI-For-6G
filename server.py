class Server(object):
    def __init__(self, algorithm):
        self.algorithm = algorithm
    
    def aggregate(self, inputs: dict):
        self.algorithm.aggregate(inputs)
        
    # def evaluate(self, data_obj, cent_x, cent_y, avg_model, all_model, device, tst_perf_sel, trn_perf_sel, tst_perf_all, trn_perf_all, t):
    #     self.algorithm.evaluate(data_obj, cent_x, cent_y, avg_model, all_model, device, tst_perf_sel, trn_perf_sel, tst_perf_all, trn_perf_all, t)