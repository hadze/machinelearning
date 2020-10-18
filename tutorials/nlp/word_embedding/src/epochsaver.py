from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile(f'{self.path_prefix}_epoch{self.epoch}.model')
        model.save(output_path)
        self.epoch += 1