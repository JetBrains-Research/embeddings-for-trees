from typing import Dict

from utils.metrics import calculate_metrics


class LearningInfo:
    def __init__(self):
        self.loss = 0.0
        self.batch_processed = 0
        self.lr = 0
        self.statistics = {
            'true_positive': 0,
            'false_positive': 0,
            'false_negative': 0
        }

    def accumulate_info(self, batch_info: Dict) -> None:
        self.loss += batch_info['loss']
        self.lr = batch_info.get('learning_rate', 0)
        self.batch_processed += 1
        for statistic in ['true_positive', 'false_positive', 'false_negative']:
            self.statistics[statistic] += batch_info['statistics'][statistic]

    def get_state_dict(self) -> Dict:
        loss = self.loss / self.batch_processed if self.batch_processed > 0 else 0.0
        state_dict = {
            'loss': loss
        }
        if self.lr != 0:
            state_dict['learning_rate'] = self.lr
        state_dict.update(calculate_metrics(self.statistics))
        return state_dict
