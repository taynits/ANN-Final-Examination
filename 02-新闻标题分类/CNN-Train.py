import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
from tensorboardX import SummaryWriter
from torchtext import data
import torch.autograd as autograd
from models.CNN import TextCnn
import  logging
logging.basicConfig(filename='Logs/CNN.log', level=logging.INFO)

def Eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        feature = feature.to(device)
        target = target.to(device)

        output = model(feature)
        loss = loss_fn(output, target)

        avg_loss += loss.data
        corrects += (torch.max(output, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * float(corrects) / size
    print('Evaluation - loss: {:.4f}  acc: {:.2f}% ({}/{})'.format(avg_loss, accuracy, corrects, size))
    logging.info('Evaluation - loss: {:.4f}  acc: {:.2f}% ({}/{})'.format(avg_loss, accuracy, corrects, size))
    return accuracy, avg_loss


# 如果有GPU就在GPU上运行
print(f'torch.cuda.is_available:{torch.cuda.is_available()}')
logging.info(f'torch.cuda.is_available:{torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.001
epochs = 10
batch_size = 128
log_interval = 100
save_interval = 1000
eval_interval = 200

dropout = 0.5
embed_dim = 128
kernel_num = 10
kernel_sizes = [3,4,5]

# ---------------数据处理---------------------
# 定义分词器
def tokenize(text):
    return [word for word in jieba.cut(text) if word.strip()]

text_field = data.Field(lower=True, tokenize = tokenize)
label_field = data.Field(sequential=False)
fields = [('text', text_field), ('label', label_field)]
train_dataset, eval_dataset = data.TabularDataset.splits(
    path = './data/', format = 'tsv', skip_header = False,
    train = 'train.tsv', test = 'dev.tsv', fields = fields
)
text_field.build_vocab(train_dataset, eval_dataset, min_freq = 3, max_size = 50000)
label_field.build_vocab(train_dataset, eval_dataset)
train_loader, eval_loader = data.Iterator.splits(
    (train_dataset, eval_dataset),
    batch_sizes = (batch_size, batch_size), sort_key = lambda x: len(x.text))

embed_num = len(text_field.vocab)
class_num = len(label_field.vocab) - 1

model = TextCnn(embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout)
model = model.to(device)
print(model)
logging.info(model)
print((embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout) )
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# ------------train and eval--------------
print("----------train and eval----------")
logging.info("----------train and eval----------")
writer = SummaryWriter("Logs/logs_CNN")
model.train()
steps = 0
best_acc = 0.0
for epoch in range(epochs):
    for batch in train_loader:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        feature = feature.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(feature)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        steps += 1
        if steps % log_interval == 0:
            corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * float(corrects) / batch.batch_size
            print('Train - Batch[{}] - loss: {:.4f}  acc: {:.2f}%({}/{})'.format(steps, loss.data, accuracy, corrects, batch.batch_size))
            logging.info('Train - Batch[{}] - loss: {:.4f}  acc: {:.2f}%({}/{})'.format(steps, loss.data, accuracy, corrects, batch.batch_size))
            writer.add_scalar('Training Accuracy', accuracy, steps)
            writer.add_scalar('Training Loss', loss.data, steps)

        if steps % eval_interval == 0:
            dev_acc, dev_loss = Eval(eval_loader, model)
            writer.add_scalar('Validation Accuracy', dev_acc, steps)
            writer.add_scalar('Validation Loss', dev_loss, steps)
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(model.state_dict(), "results/model_CNN_best.pth")
                print("Best CNN model saved.")
                logging.info("Best CNN model saved.")
            print('\n')
writer.close()
print("best model - accuracy: {:.2f}%".format(best_acc))
logging.info("best model - accuracy: {:.2f}%".format(best_acc))
