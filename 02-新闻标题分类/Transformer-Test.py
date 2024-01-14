import torch
import torch.nn as nn
import jieba
from torchtext import data
import torch.autograd as autograd
from models.Transformer import MyNet

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(text):
    return [word for word in jieba.cut(text) if word.strip()]

text_field = data.Field(lower=True, tokenize = tokenize)
label_field = data.Field(sequential=False)
fields = [('text', text_field), ('label', label_field)]
train_dataset, eval_dataset = data.TabularDataset.splits(
    path = './data/', format = 'tsv', skip_header = False,
    train = 'train.tsv', test = 'dev.tsv', fields = fields
)
text_field.build_vocab(train_dataset, eval_dataset, min_freq = 5, max_size = 50000)
label_field.build_vocab(train_dataset, eval_dataset)

maxlen = max(len(example.text) for example in train_dataset.examples)
vocab_size = len(text_field.vocab)
embed_dim = 128 # 嵌入维度
feed_dim = 128 # feed-forward 网络的维度
num_heads = 2 # 注意力头的数量
model = MyNet(maxlen, vocab_size, embed_dim, feed_dim, num_heads)
model = model.to(device)
print(model)
test_path = 'results/model_TRANS_best.pth'
model.load_state_dict(torch.load(test_path))
model.to(device)
model.eval()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print ("Model parameters: " + str(pytorch_total_params))

with open('./data/test.txt', 'r', encoding='utf-8') as file:
    test_lines = file.readlines()

with open('results/Transformer_test_result.txt', 'w', encoding='utf-8') as result_file:
    for text in test_lines:
        label = Predict(text, model, text_field, label_field)
        result = str(label) + " | " + text
        result_file.write(result)

print("测试结果保存在: results/Transformer_test_result.txt")
