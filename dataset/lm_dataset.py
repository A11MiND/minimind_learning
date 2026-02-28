import json

from torch.utils.data import Dataset
import torch
import os

#是指tokenizer的并行化，设置为false可以避免在多线程环境下出现问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 写dataset类，继承torch.utils.data.Dataset
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

# 实现dataset内定的方法
    def load_data(self,path):
        samples=[]
        with open(path, "r", encoding="utf-8") as f:
            for line_num,line in enumerate(f,1):
                data=json.loads(line.strip())
                samples.append(data)
        return samples

# __len__
    def __len__(self):
        return len(self.samples)
    
# __getitem__
    def __getitem__(self, index):
        sample=self.samples[index]

        encoding=self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        #(max_length,)的张量，去掉batch维度
        input_ids=encoding["input_ids"].squeeze()

        # [1, ,1, 1, 0 , 0 ]
        loss_mask=input_ids!=self.tokenizer.pad_token_id

        # 自回归
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)

        loss_mask=torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask

