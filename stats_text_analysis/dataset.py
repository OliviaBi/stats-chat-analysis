from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, messages, dates):
        self.messages = messages
        self.dates = dates

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return {"messages": self.messages[idx], "dates": self.dates[idx]}


class TwoChatsDataset(Dataset):
    def __init__(self, messages, dates, sender):
        self.messages = messages
        self.dates = dates
        self.sender = sender

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return {
            "messages": self.messages[idx],
            "dates": self.dates[idx],
            "sender": self.sender[idx],
        }
        
