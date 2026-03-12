import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from .dataset_loader import load_dataset
from .job_model import JobModel


class JobDataset(Dataset):

    def __init__(self, texts, tokenizer):

        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        tokens = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return tokens["input_ids"].squeeze()


def train():

    texts = load_dataset("dataset/jobs.csv")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = JobDataset(texts, tokenizer)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training on:", device)

    model = JobModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        progress = tqdm(dataloader)

        for batch in progress:

            batch = batch.to(device)

            output = model(batch)

            loss = output.mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            progress.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "saved_model/job_model.pt")

    print("Model saved to saved_model/job_model.pt")