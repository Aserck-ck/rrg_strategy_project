import sys
import gc
print(sys.executable)

from rrg_stock_a.models.module import *
from dataset import*

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU缓存已清理")
    except Exception as e:
        print(f"缓存清理失败: {e}")

time_window = 200
predict_window = 5
batch_size = 12
epochs = 20
patience = 8
lr = 1e-4

continue_training = False
# model_path='./enhanced_lstm_models/sw_level2_enlstm.pth'
model_path='./enhanced_lstm_models/sw_level2_gru_e.pth'
if continue_training:
    model = torch.load(model_path, weights_only=False, map_location=device)
    # best_val_loss = 0.104679
    best_val_loss = 0.108
else:
    model = GRUModel(4, 4, 512,0.25)
    best_val_loss = float('inf')

loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.9),
        weight_decay=1e-3,
    )



model = model.to(device)
train_loader, test_loader, train_dataset, test_dataset = prepare_sw2_dataloaders(time_window, predict_window, batch_size)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.03, div_factor=10
    )


rollback_cnt=0

for idx in range(epochs):
    model.train()
    train_loss=0.0


    # print(f"start training epoch:{i}, train_sample_num:{len(train_loader)}, test_sample_num:{len(test_loader)}")
    train_dataset.set_epoch_seed(1145*idx+1)
    test_dataset.set_epoch_seed(1919*idx+1)
    for dataset in train_loader:
        dataset = dataset.squeeze(0).float()
        inputs = dataset[:, :time_window, :]
        targets = dataset[:, time_window:, :]
        inputs = inputs.to(device)
        targets = targets.to(device)
        pre_data = None
        single_loss=0
        for i in range(predict_window):
            if pre_data is not None:
                inputs = torch.cat([inputs,pre_data],dim=1)
            target = targets[:,i,:].clone()
            target = target.unsqueeze(1)
            preds = model(inputs)
            single_loss += loss_function(preds, target)*np.exp(-i)
            pre_data=preds

        optimizer.zero_grad()
        single_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        scheduler.step()

        train_loss+= single_loss.item()

        del inputs, targets, preds




    print('epoch: %d, avg train loss: %.6f.' % (idx, train_loss / len(train_loader)), end='; ')

    # Test whether the model is generalizable to the test set
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for test_data in test_loader:
            test_data=test_data.squeeze(0).float()
            test_in = test_data[:, :time_window, :]
            test_tar = test_data[:, time_window:, :]
            test_input = test_in.to(device)
            test_target = test_tar.to(device)
            pre_data = None
            single_loss = 0
            for i in range(predict_window):
                if pre_data is not None:
                    test_input = torch.cat([test_input, pre_data], dim=1)
                target = test_target[:, i, :].clone()
                target = target.unsqueeze(1)
                preds = model(test_input)
                single_loss += loss_function(preds, target) * np.exp(-i)
                pre_data = preds

            # test_target=test_target.mean(dim=1)
            # test_target=test_target.unsqueeze(1)

            # y_pred = model(test_input)
            total_loss += single_loss
            del test_input, test_target, preds

    val_loss = total_loss.item() / len(test_loader)

    print('epoch: %d, avg test loss: %.6f. ' % (idx, val_loss), end='\n')
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    if val_loss < best_val_loss:
        rollback_cnt=0
        torch.save(model, model_path)
        best_val_loss = val_loss
        print(f"模型已存储")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                print("GPU缓存已清理")
            except Exception as e:
                print(f"缓存清理失败: {e}")
    else:
        rollback_cnt +=1
        print(f"回滚计数: {rollback_cnt}")
        if rollback_cnt == patience:
            model = torch.load('./enhanced_lstm_models/sw_level2_gru.pth', weights_only=False, map_location=device)
            rollback_cnt=0
            print(f"模型回滚成功")

