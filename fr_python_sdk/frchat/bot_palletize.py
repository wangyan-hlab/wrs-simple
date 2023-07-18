import yaml
from frchat.bot import FRChatBot

class FRChatBotPalletize(FRChatBot):
    """
        A ChatBot generating robot palletization programs

        Author: wangyan
        Date: 2023/05/22
    """
    
    def __init__(self, messages, temperature=0.3, model="gpt-3.5-turbo", history_num_to_del=0) -> None:
        super().__init__(messages, temperature, model, history_num_to_del)
        self.messages = messages

    def read_config(self, file):
        """
            Reading palletization params from a YAML file
        """
        with open(file, "rb") as f:
            params = yaml.safe_load(f)
        
        return params
        