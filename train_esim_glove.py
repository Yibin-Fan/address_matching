import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from esim_model import ESIM_Model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextMatchDataset(Dataset):
    def __init__(self, text1_path, text2_path, label_path, max_len=128):
        self.max_len = max_len
        self.text1_data = self._load_text_data(text1_path)
        self.text2_data = self._load_text_data(text2_path)
        with open(label_path, 'r') as f:
            self.labels = np.array([float(line.strip()) for line in f])

    def _load_text_data(self, file_path):
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                nums = [int(x) for x in line.strip().split()]
                if len(nums) > self.max_len:
                    nums = nums[:self.max_len]
                else:
                    nums = nums + [0] * (self.max_len - len(nums))
                sequences.append(nums)
        return np.array(sequences, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'addr1': torch.LongTensor(self.text1_data[idx]),
            'addr2': torch.LongTensor(self.text2_data[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }


def load_glove_embeddings(vocab_path, embeddings_path, embedding_dim):
    """加载二进制词向量文件并创建嵌入矩阵"""
    vocab = {}
    embeddings_vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word = line.split()[0]
            vocab[word] = i
            embeddings_vocab[word] = i

    vocab_size = len(vocab)
    embedding_matrix = np.fromfile(embeddings_path, dtype=np.float32)
    embedding_matrix = embedding_matrix.reshape(vocab_size, embedding_dim)

    embedding_matrix_final = np.zeros((vocab_size, embedding_dim))
    for word, i in vocab.items():
        if word in embeddings_vocab:
            embedding_matrix_final[i] = embedding_matrix[embeddings_vocab[word]]
        else:
            embedding_matrix_final[i] = np.random.uniform(low=-0.05, high=0.05, size=(embedding_dim,))

    return torch.tensor(embedding_matrix_final, dtype=torch.float32), vocab_size


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            text1 = batch['addr1'].to(device)
            text2 = batch['addr2'].to(device)
            labels = batch['label'].to(device)

            outputs = model(text1, text2)
            predictions.extend((outputs > 0.5).float().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_f1 = 0
    # Initialize history dictionary to store metrics
    history = {
        'loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            text1 = batch['addr1'].to(device)
            text2 = batch['addr2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(text1, text2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)

        # Store metrics in history
        history['loss'].append(avg_loss)
        history['precision'].append(val_precision)
        history['recall'].append(val_recall)
        history['f1'].append(val_f1)

        # Log all metrics
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]')
        logger.info(f'Training Loss: {avg_loss:.4f}')
        logger.info(f'Validation Precision: {val_precision:.4f}')
        logger.info(f'Validation Recall: {val_recall:.4f}')
        logger.info(f'Validation F1: {val_f1:.4f}')

        # Save model if F1 score improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'precision': val_precision,
                'recall': val_recall,
            }, save_path)
            logger.info("best model found and saved")

        logger.info('-' * 50)

    # Plot training history
    plot_training_history(history, num_epochs)

    return history

def plot_training_history(history, num_epochs):
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)

    # Plot all metrics
    plt.plot(epochs, history['precision'], 'b-', label='Precision')
    plt.plot(epochs, history['recall'], 'g-', label='Recall')
    plt.plot(epochs, history['f1'], 'r-', label='F1')
    plt.plot(epochs, history['loss'], 'y-', label='Loss')

    plt.title('Training Metrics History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('results/word2vec/diagram.png')
    plt.close()

def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    MAX_LEN = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EMBEDDING_DIM = 100  # 根据你生成的vectors.bin的维度调整

    logger.info(f'Using device: {DEVICE}')

    # Load GloVe embeddings
    embedding_matrix, vocab_size = load_glove_embeddings(
        vocab_path='results/glove/vocab.txt',
        embeddings_path='results/glove/vectors.bin',
        embedding_dim=EMBEDDING_DIM
    )

    # Load datasets
    train_dataset = TextMatchDataset(
        'data/glove_dataset/train/addr1_tokenized.txt',
        'data/glove_dataset/train/addr2_tokenized.txt',
        'data/glove_dataset/train/labels.txt',
        max_len=MAX_LEN
    )

    val_dataset = TextMatchDataset(
        'data/glove_dataset/test/addr1_tokenized.txt',
        'data/glove_dataset/test/addr2_tokenized.txt',
        'data/glove_dataset/test/labels.txt',
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = ESIM_Model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        word2vec_matrix=embedding_matrix,
        max_sequence_length=MAX_LEN
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_path='results/glove/best_esim_model.pth'
    )

    logger.info("Training completed")

if __name__ == '__main__':
    main()