import time

def generate_log_dir(config):
    dataset = config.source
    current_time = time.strftime('%Y%m%d%H%M')
    batch_size = config.batch_size
    learning_rate = config.lr
    seq_len = config.seq_len
    horizon = config.horizon
    logdir = './logs/%s/%s_lr_%f_sl_%d_h_%d_bs_%d/' % (
        dataset, current_time, learning_rate, seq_len, horizon, batch_size)
    return logdir

if __name__ == '__main__':
    current_time = time.strftime('%Y%m%d%H%M')
    print(current_time)

