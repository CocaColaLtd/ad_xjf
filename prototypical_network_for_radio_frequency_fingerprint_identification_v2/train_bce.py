# coding=utf-8
from lib.utils.dataset.RFF_dataset_h5 import RFFDataset
from lib.model.resnet import resnet
from lib.model.resnet_octave_conv_lswt import resnet_octave_conv
from lib.utils.parser.parser_util import get_parser
from torch.utils.data import DataLoader
import logging
from lib.utils.timer import timer
import shutil
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
from test import test
from t_sne import plot_t_sne

torch.backends.cudnn.enabled = False
print(torch.version.cuda)
print(torch.cuda.get_device_capability())


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    datapath = "/home/xjf/data/add_noise_4096_transient_for_training_with_10_devices"
    dataset = RFFDataset(mode=mode, root=datapath)
    dataset_num = len(dataset.x)
    print(mode, 'dataset_num', dataset_num)
    logging.info('%s dataset_num:%d' % (mode, dataset_num))
    n_classes = len(np.unique(dataset.y))
    return dataset, n_classes


def init_dataloader(opt, mode):
    dataset, n_classes = init_dataset(opt, mode)
    print(f"{mode} dataset size: {len(dataset)}")
    
    # 使用标准的DataLoader，而不是PrototypicalBatchSampler
    if mode == 'training':
        shuffle = True
        batch_size = opt.batch_size_train
    else:
        shuffle = False
        batch_size = opt.batch_size_val
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    return dataloader, n_classes


def init_model(opt, n_classes):
    '''
    Initialize the model
    '''
    device = f'cuda:{opt.cuda}' if torch.cuda.is_available() else 'cpu'
    
    if opt.model == 'resnet':
        model = resnet(num_classes=n_classes).to(device)
    elif opt.model == 'resnet_octave_conv_lswt':
        model = resnet_octave_conv(num_classes=n_classes).to(device)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.6fM" % (total/1e6))
    print(model)
    logging.info(model)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate,
                            weight_decay=opt.weight_decay)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def train(opt, tr_dataloader, model, optim, lr_scheduler, experiments_root, val_dataloader=None):
    '''
    Train the model with standard classification

    '''
    from lib.model.resnet_octave_conv_lswt import octave_conv ,residual_stack
    device = f'cuda:{opt.cuda}' if torch.cuda.is_available() else 'cpu'
    
    if opt.model == 'resnet':
        best_model_net_path = 'lib/model/resnet.py'
    elif opt.model == 'resnet_octave_conv_lswt':
        best_model_net_path = 'lib/model/resnet_octave_conv_lswt.py'
    
    shutil.copy(best_model_net_path, experiments_root)
    best_model_path = os.path.join(experiments_root, 'best_model.pth')
    last_model_path = os.path.join(experiments_root, 'last_model.pth')
    
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    train_loss = []
    train_ortho = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    
    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        
        # 训练阶段
        model.train()
        epoch_train_loss = []
        epoch_train_acc = []
        
        for batch_idx, (x, y) in enumerate(tqdm(tr_dataloader)):
            x, y = x.to(device), y.to(device)
            
            optim.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            # 收集所有嵌套在residual_stack中的octave_conv模块的正交损失
            ortho_loss = 0.0
            for module in model.modules():
                if isinstance(module, residual_stack) and hasattr(module.rs_conv1, 'ortho_loss'):
                    ortho_loss += module.rs_conv1.ortho_loss
                elif isinstance(module, octave_conv) and hasattr(module, 'ortho_loss'):
                    ortho_loss += module.ortho_loss
 
            # 总损失计算和反向传播保持不变
            # total_loss = loss + opt.ortho_lambda * ortho_loss
            total_ortho = sum([module.ortho_loss for module in model.modules() 
                      if hasattr(module, 'ortho_loss')])
            total_loss = loss + total_ortho  # loss是张量，total_ortho也是张量
    
            # 记录时转换为float
            train_ortho.append(total_ortho.detach().cpu().item())             
            total_loss.backward()
            # loss.backward()
            optim.step()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
            
            epoch_train_loss.append(total_loss.item())
            epoch_train_acc.append(accuracy)
        
        avg_loss_train = np.mean(epoch_train_loss)
        avg_acc_train = np.mean(epoch_train_acc)
        train_loss.extend(epoch_train_loss)
        train_acc.extend(epoch_train_acc)
        
        learning_rate = optim.state_dict()['param_groups'][0]['lr']
        avg_ortho = np.mean(train_ortho[-opt.iterations:]) if train_ortho else 0.0
        print('Avg Train Loss: {:.4f}, Ortho Loss: {:.4f}, Avg Train Acc: {:.4f}, LR: {:.6f}'.format(
            avg_loss_train, avg_ortho, avg_acc_train, learning_rate))
        print('Avg Train Loss: {}, Avg Train Acc: {}, learning_rate:{}'.format(avg_loss_train, avg_acc_train, learning_rate))       
        lr_scheduler.step()
        
        # 验证阶段
        if val_dataloader is not None:
            model.eval()
            epoch_val_loss = []
            epoch_val_acc = []
            
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(val_dataloader):
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    total_loss = loss + total_ortho
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == y).sum().item()
                    accuracy = correct / y.size(0)
                    
                    epoch_val_loss.append(total_loss.item())
                    epoch_val_acc.append(accuracy)
            
            avg_loss_val = np.mean(epoch_val_loss)
            avg_acc_val = np.mean(epoch_val_acc)
            val_loss.extend(epoch_val_loss)
            val_acc.extend(epoch_val_acc)
            
            postfix = ' (Best)' if avg_acc_val >= best_acc else ' (Best: {:.4f})'.format(best_acc)
            print('Avg Val Loss: {:.4f}, Avg Val Acc: {:.4f}{}'.format(
                avg_loss_val, avg_acc_val, postfix))
            
            # 记录日志
            if avg_acc_val >= best_acc:
                logging.info(
                    'Epoch: %d, Avg Train Loss: %.6f, Avg Train Acc: %.6f, Ortho Loss:  %.6f, Avg Val Loss: %.6f, Avg Val Acc: %.6f(best), LR: %.6f' % (
                        epoch, avg_loss_train, avg_acc_train,avg_ortho, avg_loss_val, avg_acc_val, learning_rate))
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc_val
            else:
                logging.info(
                    'Epoch: %d, Avg Train Loss: %.6f, Avg Train Acc: %.6f,Ortho Loss:  %.6f, Avg Val Loss: %.6f, Avg Val Acc: %.6f(best:%.6f), LR: %.6f' % (
                        epoch, avg_loss_train, avg_acc_train,avg_ortho, avg_loss_val, avg_acc_val, best_acc, learning_rate))
            if avg_acc_val >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc_val
        # 定期绘制t-SNE图
        if epoch % 10 == 0:
            plot_t_sne(opt, model, experiment_root=experiments_root, epoch=epoch)
    
    torch.save(model.state_dict(), last_model_path)
    
    return best_model_path, best_acc, train_loss, train_acc, val_loss, val_acc


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    
    # 添加标准分类任务所需的参数（如果parser中没有的话）
    if not hasattr(options, 'batch_size_train'):
        options.batch_size_train = 64
    if not hasattr(options, 'batch_size_val'):
        options.batch_size_val = 64
    if not hasattr(options, 'num_workers'):
        options.num_workers = 4
    if not hasattr(options, 'weight_decay'):
        options.weight_decay = 1e-4
    
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
    
    time_info = timer.get_datetime_millisecond_with_hyphen(None)
    experiments_root = os.path.join(options.experiment_root, time_info)
    if not os.path.exists(experiments_root):
        os.makedirs(experiments_root)
    
    logging.basicConfig(format='%(asctime)s -  %(levelname)s: %(message)s',
                        level=logging.INFO,
                        filename=os.path.join(experiments_root, 'epoch_log.txt'),
                        filemode='w')
    
    device = f'cuda:{options.cuda}' if torch.cuda.is_available() else 'cpu'
    print(f'train_device:{device}')
    logging.info(f'train_device:{device}')
    
    init_seed(options)
    
    # 初始化数据加载器
    tr_dataloader, n_classes = init_dataloader(options, 'training')
    val_dataloader, _ = init_dataloader(options, 'validing')
    
    # 初始化模型、优化器和学习率调度器
    model = init_model(options, n_classes).to(device)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    
    # 训练模型
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                experiments_root=experiments_root)
    
    best_model_path, best_acc, train_loss, train_acc, val_loss, val_acc = res
    
    print('Testing with last model..')
    logging.info('Testing with last model..')
    test(opt=options, model=model)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(best_model_path))
    print('Testing with best model..')
    logging.info('Testing with best model..')
    test(opt=options, model=model)


if __name__ == '__main__':
    main()