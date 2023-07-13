from torch.utils.data import Dataset


class ClsDataset(Dataset):
    def __init__(self, features):
        self.input_ids = features[0]
        self.attention_masks = features[1]
        self.label_ids = features[2]
        self.num = len(self.input_ids)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_masks[item], self.label_ids[item]