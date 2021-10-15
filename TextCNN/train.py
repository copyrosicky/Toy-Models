'''
TEXT CNN 的模型训练和评估
'''
import os
import sys
import time

import torch
import torch.autograd as autograd
import torch.nn.functional as F


def evaluate(data_iter, model, args):
    # 使用正确率来评估模型
    model.eval()
    corrects, avg_loss = 0, 0
    loss_function = F.cross_entropy()
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = loss_function(logit, target, size_average=False)
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def train(model, train_batch_size, train_iter,learning_rate, weight_decay, args, Epochs):

    # 指定gpu运行
    if torch.cuda.is_available():
        model.cuda()
    # 交叉熵函数
    loss_function = F.cross_entropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_batch_size, gamma=0.8)  # 每过step_size个epoch，做一次更新

    best_accuracy = 0.0
    for epoch in range(Epochs):
        for step, x in enumerate(train_iter):
            train_loss_sum = 0.0
            start_time = time.time()
            model.train()
            feature, target = x.text, x.label
            feature.t_(), target.sub_(1)  # batch first, index align

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = loss_function(logit, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_iter):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch + 1, step + 1, len(train_iter), train_loss_sum / (step + 1), time.time() - start_time))

        scheduler.step()
        accuracy = evaluate(model)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs('./save_model', exist_ok=True)
            torch.save(model.state_dict(), './save_model/textcnn.bin')



def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


