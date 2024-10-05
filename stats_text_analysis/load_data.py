from torch.utils.data import DataLoader
from .dataset import ChatDataset, TwoChatsDataset
from .utils import get_chats_by_name, get_chats_by_sender
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data(chats_data, contact, sender, length=1280 * 4, shuffle=False):
    if sender != "both":
        chats_with_contact = get_chats_by_name(chats_data, contact)
        chats_messages = chats_with_contact.get("messages")
        chats = get_chats_by_sender(chats_messages, sender, limit=length)
        chats_dataset = ChatDataset(chats["text"], chats["date"])
        chats_dataloader = DataLoader(chats_dataset, batch_size=64, shuffle=shuffle)
        return chats_dataloader
    else:
        chats_with_contact = get_chats_by_name(chats_data, contact)
        chats_messages = chats_with_contact.get("messages")
        chats = get_chats_by_sender(chats_messages, sender="both", limit=length)
        two_chats_dataset = TwoChatsDataset(
            chats["text"], chats["date"], chats["sender"]
        )
        two_chats_dataloader = DataLoader(
            two_chats_dataset, batch_size=64, shuffle=shuffle
        )
        return two_chats_dataloader