from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f'Epoch #{self.epoch} start')

    def on_epoch_end(self, model):
        print(f'Epoch #{self.epoch} end')
        self.epoch += 1

