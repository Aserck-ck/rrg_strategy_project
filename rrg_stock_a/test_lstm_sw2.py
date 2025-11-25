import sys
import gc
print(sys.executable)

from dataset import*



def generate_prediction(time_window=100,predict_window = 5, samples=10, model_path='./enhanced_lstm_models/sw_level2_lstm.pth',start = 0, end =-1):

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

    window = time_window+predict_window

    d_in=4


    model = torch.load(model_path, weights_only=False, map_location=device)
    model = model.to(device)

    test_list, test_dataset, scaler_ratio, scaler_momentum = prepare_testdata(time_window, predict_window)

    model.eval()

    sample_num=500-window+1
    if end<0:
        end=sample_num+1+end
    if start<0:
        start = sample_num+start
    samples = min(len(test_list[0].columns),samples)
    columns = test_list[0].columns[:samples]
    indexs=test_list[0].index[time_window:]
    pred_df = [pd.DataFrame(index=indexs,columns=columns, data=np.zeros((len(indexs), len(columns)))) for _ in range(d_in)]
    target_df = [pd.DataFrame(index=indexs,columns=columns, data=np.zeros((len(indexs), len(columns)))) for _ in range(d_in)]
    with torch.no_grad():
        for col in columns:
            pred_cnt = np.zeros(len(indexs))
            for i in range(start,end):
                test_data = test_dataset[col][i:i+window].clone()
                test_in = test_data[:time_window, :]
                test_tar = test_data[time_window:, :]
                test_input = test_in.to(device).unsqueeze(0)
                test_target = test_tar.to(device).unsqueeze(0)
                # test_target = test_target.mean(dim=1)
                # test_target = test_target.unsqueeze(1)
                y_pred = model(test_input)

                pred_cnt[i:i+predict_window]+=1
                for j in range(y_pred.shape[-1]):
                    pred_df[j][col].iloc[i:i+predict_window]+=np.array(y_pred[0,:,j].cpu())
                    target_df[j][col].iloc[i:i+predict_window]+=np.array(test_target[0,:,j].cpu())

            for j in range(d_in):
                pred_df[j][col]/=pred_cnt
                target_df[j][col]/=pred_cnt


    for i in range(2):
        pred_df[i]=scaler_ratio.inverse_transform(pred_df[i])
        target_df[i]=scaler_ratio.inverse_transform(target_df[i])
        pred_df[i]=pd.DataFrame(index=indexs,columns=columns,data=pred_df[i])
        target_df[i]=pd.DataFrame(index=indexs,columns=columns,data=target_df[i])

    for i in range(2,4):
        pred_df[i] = scaler_momentum.inverse_transform(pred_df[i])
        target_df[i] = scaler_momentum.inverse_transform(target_df[i])
        pred_df[i] = pd.DataFrame(index=indexs, columns=columns, data=pred_df[i])
        target_df[i] = pd.DataFrame(index=indexs, columns=columns, data=target_df[i])

    return pred_df, target_df


# pred_df, target_df = generate_prediction(100,5,5,'./enhanced_lstm_models/sw_level2_gru.pth')
