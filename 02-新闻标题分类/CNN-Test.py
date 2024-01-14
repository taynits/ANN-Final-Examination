import jieba
from torch import autograd
from models.CNN import TextCnn
import torch
from torchtext import data

def tokenize(text):
    return [word for word in jieba.cut(text) if word.strip()]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Predict(text, model, text_field, label_feild):
    assert isinstance(text, str)

    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    x = x.to(device)

    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data + 1]


embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout = \
    50002, 128, 14, 10, [3, 4, 5], 0.5
model = TextCnn(embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout)
model = model.to(device)
test_path = 'results/model_CNN_best.pth'
model.load_state_dict(torch.load(test_path))
model.to(device)
model.eval()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print ("Model parameters: " + str(pytorch_total_params))

text_field = data.Field(lower=True, tokenize = tokenize)
label_field = data.Field(sequential=False)
fields = [('text', text_field), ('label', label_field)]
train_dataset, test_dataset = data.TabularDataset.splits(
    path = '/home/dell/人工神经网络/02-新闻标题分类/data/', format = 'tsv', skip_header = False,
    train = 'train.tsv', test = 'dev.tsv', fields = fields
)
text_field.build_vocab(train_dataset, test_dataset, min_freq = 5, max_size = 50000)
label_field.build_vocab(train_dataset, test_dataset)

with open('./data/test.txt', 'r', encoding='utf-8') as file:
    test_lines = file.readlines()

with open('results/CNN_test_result.txt', 'w', encoding='utf-8') as result_file:
    for text in test_lines:
        label = Predict(text, model, text_field, label_field)
        result = str(label) + " | " + text
        result_file.write(result)

print("测试结果保存在: results/CNN_test_result.txt")
