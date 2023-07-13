from tqdm import tqdm
from configs import *
from torch import optim
from models import BertCls
from BertLSTMBayesClassifier import BertLSTMBayesClassifier
from dataset import ClsDataset
from transformers import BertTokenizer
from preprocess import get_tag2idx, load_comments_and_tags, preprocess_data
from torch.utils.data import DataLoader
from metric import *
import matplotlib.pyplot as plt


def accuracy(model, dataloader):
    model.eval()
    num_acc, num_samples = 0, 0
    for input_ids, attention_mask, tags in dataloader:
        input_ids, attention_mask, tags = input_ids.to(device), attention_mask.to(device), tags.to(device)
        output = model(input_ids, attention_mask)
        num_acc += torch.sum(output.argmax(dim=-1) == tags)
        num_samples += len(input_ids)
    return float(num_acc / num_samples)

def score(model, dataloader):
    model.eval()
    scores = []
    true_y = []
    for input_ids, attention_mask, tags in dataloader:
        input_ids, attention_mask, tags = input_ids.to(device), attention_mask.to(device), tags.to(device)
        output = model(input_ids, attention_mask)
        scores.extend(output.tolist())
        true_y.extend(tags.tolist())
        #pred_y.extend(output.argmax(dim=-1).tolist())

    return true_y,scores


def test_model():
    tag2idx, idx2tag = get_tag2idx(train_file)
    num_classes = len(tag2idx)
    # model = BertCls(num_classes)
    #model = BertLSTMBayesClassifier(num_classes, num_hiddens, lstm_num_layers,bert_model_name,n_samples)
    model = SimpleBert(num_classes)
    model.load_state_dict(torch.load(f'outputs/{best_model_name}'))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    test_dataset = ClsDataset(preprocess_data(test_file, tokenizer, max_seq_len))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    tags,scores= score(model, test_dataloader)
    results = evaluate_all(tags,scores)
    
    print(results) 

def train_model():
    tag2idx, idx2tag = get_tag2idx(train_file)
    num_classes = len(tag2idx)

    model = SimpleBert(num_classes)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay = weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    print('start loading data.')
    train_dataset = ClsDataset(preprocess_data(train_file, tokenizer, max_seq_len))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = ClsDataset(preprocess_data(val_file, tokenizer, max_seq_len))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    print('finish loading data.')

    print('start training model.')
    best_val_acc = 0
    num = range(1,num_epochs+1)
    loss_num = []
    v_acc = []
    t_acc = []
    val_loss = []
    v_loss = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        for input_ids, attention_mask, tags in tqdm(train_dataloader):
            input_ids, attention_mask, tags = input_ids.to(device), attention_mask.to(device), tags.to(device)
            output = model(input_ids, attention_mask)
            optimizer.zero_grad()
            loss = criterion(output, tags)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
        for input_ids, attention_mask, tags in tqdm(val_dataloader):
            input_ids, attention_mask, tags = input_ids.to(device), attention_mask.to(device), tags.to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, tags)
            loss.backward()
            val_loss.append(loss.item())
        val_acc = accuracy(model, val_dataloader)
        train_acc = accuracy(model,train_dataloader)
        t_acc.append(train_acc)
        v_acc.append(val_acc)
        val_acc = accuracy(model, val_dataloader)
        train_acc = accuracy(model, train_dataloader)
        val_acc = accuracy(model, val_dataloader)
        train_acc = accuracy(model, train_dataloader)
        print(f'train acc {train_acc}')
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            model.to('cpu')
            torch.save(model.state_dict(), f'outputs/{best_model_name}')
            print(f'best val acc {best_val_acc}, save model to outputs/{best_model_name}')
            model.to(device)
        a = sum(train_loss) / len(train_loss)
        loss_val = sum(val_loss) / len(val_loss)
        loss_num.append(a)
        v_loss.append(loss_val)
        print(f'epoch {epoch + 1}, train loss {sum(train_loss) / len(train_loss):.4f}')
    test_model()
    print(f'val best acc { best_val_acc}')
    print(f'val best acc { best_val_acc}')
    print((loss_num))
    print((v_loss))
    print((t_acc))
    print((v_acc))
    print((num))

    plt.plot(num, t_acc, "g*-", label='train_acc')
    plt.plot(num, v_acc, "b*-", label='val_acc')
    plt.xlabel('num_epoch')
    plt.title('Bert', pad=15, size=15)
    plt.legend()
    plt.savefig('output_image/acc/avlinear3_acc.png')
    plt.clf()
    plt.plot(num, loss_num, "k*-", label='train_loss')
    plt.plot(num, v_loss, "y*-", label='val_loss')
    plt.xlabel('num_epoch')
    plt.title('Bert', pad=15, size=15)
    plt.legend()
    plt.savefig('output_image/loss/avlinear3_loss.png')
    print('image saved')
    print('finish training model.')


train_model()