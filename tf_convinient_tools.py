import tensorflow as tf
import numpy as np
import time
import os
from datetime import datetime

def tensor_trainer(train_x,
                   x_placeholder, #入力用のplaceholder
                   train_y,
                   y_placeholder, #教師用のplaceholder
                   optimizer_node, #最適化用のノードを渡す
                   feed_dict={},#train中にgraphに与えるその他の定数を格納する辞書
                   loss_node=None, #学習時のlossを示すのに使う
                   accuracy_node=None, #学習時のaccuracyを示すのに使う
                   show_nodes = {}, #学習時に表示したいノード値を名前をkeyにして格納。表示はnp.sumでスカラー化される ex){'accuracy':accuracy_node}
                   test_x=None, #logやtestのshow_nodesを表示するのに必要
                   test_y=None,
                   test_feed_dict=None, #test中にgraphに与えるfeed_dict Noneのときはfeed_dictと同一とする
                   batch_size=128,
                   epochs=10, #このエポック数-1まで回す
                   start_epochs=0,#このエポック数から回す
                   is_log=False, #logを取る。logを取るときは必ずTrueにする
                   log_dir='./tmp_log', #tensorboard用のログを出力する先 tensorboard --logdir=　で指定する場所
                   log_interval_epochs=None, #(epoch+1)%interval==0でlogを取る Noneで10回まで取るように自動で調整
                   is_save=False, #モデルのパラメータを保存する
                   max_to_keep = 5, #saveするファイルの最大数。これ以上のファイルが保存されそうになると一番古いものを削除する
                   model_name='./noname_model', #実際に保存されるときは末尾にエポックの回数(epoch+1)がつく
                   load_model_name=None, #Noneでないときそのモデルを最初に読み込む
                   save_interval_epochs=1, #(epoch+1)%interval==0でsaveする
                   is_save_last = False, #Trueならば学習後にsaveする {model_name}-{epochs}-last で保存される
                   is_print = True, #標準出力の可否。（ただし、tensorflowのものは除く）
                   target='', #計算対象とするGPU
                   graph=None, #計算対象とするtf.Graph
                   config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) #tf.Sessionの設定。
                  ):
    '''
    Ver 1.0
    Python version >= 3.6
    TensorFlow学習用の簡単学習用の関数。面倒な学習回りを整理して実行してくれる。ログ、パラメタ保存/ロードもできる
    基本的な使用方法はコメントや変数名に合わせた情報を初期値が定義されていない変数に渡して使えばよい
    
    引数
    train_x,
    x_placeholder, #入力用のplaceholder
    train_y,
    y_placeholder, #教師用のplaceholder
    optimizer_node, #最適化用のノードを渡す
    feed_dict={},#train中にgraphに与えるその他の定数を格納する辞書
    loss_node=None, #学習時のlossを示すのに使う
    accuracy_node=None, #学習時のaccuracyを示すのに使う
    show_nodes = {}, #学習時に表示したいノード値を名前をkeyにして格納。表示はnp.sumでスカラー化される ex){'accuracy':accuracy_node}
    test_x=None, #logやtestのshow_nodesを表示するのに必要
    test_y=None,
    test_feed_dict=None, #test中にgraphに与えるfeed_dict Noneのときはfeed_dictと同一とする
    batch_size=128,
    epochs=10, #このエポック数-1まで回す
    start_epochs=0,#このエポック数から回す
    is_log=False, #logを取る。logを取るときは必ずTrueにする
    log_dir='./tmp_log', #tensorboard用のログを出力する先 tensorboard --logdir=　で指定する場所
    log_interval_epochs=None, #(epoch+1)%interval==0でlogを取る Noneで10回まで取るように自動で調整
    is_save=False, #モデルのパラメータを保存する
    max_to_keep = 5, #saveするファイルの最大数。これ以上のファイルが保存されそうになると一番古いものを削除する
    model_name='./noname_model', #実際に保存されるときは末尾にエポックの回数(epoch+1)がつく
    load_model_name=None, #Noneでないときそのモデルを最初に読み込む
    save_interval_epochs=1, #(epoch+1)%interval==0でsaveする
    is_save_last = False, #Trueならば学習後にsaveする {model_name}-{epochs}-last で保存される
    is_print = True, #標準出力の可否。（ただし、tensorflowのものは除く）
    target='', #計算対象とするGPU
    graph=None, #計算対象とするtf.Graph
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) #tf.Sessionの設定。
    
    基本的にこの関数の引数に与えたコメント通りに使えば良い。
    注意が必要なのは初期値のままでは何も保存しないため再利用してpredictするなどができない。
    これはむやみにファイルを保存することを防ぐ目的がある。
    再利用する場合はis_saveやis_save_lastをTrueにする。
    また、configは特に理由なく変更しないこと。
    configの初期値でメモリの使用量の増大を許して、最初にすべてのGPUメモリが確保されることを防いでいる。
    '''
    if load_model_name is not None:
        assert tf.gfile.Exists(load_model_name+'.index'), load_model_name+' does not exists'
    if is_log:
        assert test_x is not None or test_y is not None, 'test_x and y must have data when is_log is True'
    
    if is_print:
        print('\n|----------------------Tensor Trainer Start!----------------------|\n'+datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    start_time = time.time()
    if is_print:
        print(f'''CONDITIONS\ntrain:x:{train_x.shape},y:{train_y.shape}  ''')
        if test_x is not None:
            print('test:x:{test_x.shape},y{test_y.shape}  ')
        else:
            print('test:None')
        print(f'epochs:[{start_epochs},{epochs}]  batch_size:{batch_size}')
        print(f'LOG_SETTING\nis_log:{is_log}  log_dir:{log_dir}  log_interval_epochs:{log_interval_epochs}')
        print(f'SAVE_MODEL_SETTING\nis_save:{is_save}  model_name:{model_name}  save_interval_epochs:{save_interval_epochs}  is_save_last:{is_save_last}')
        print(f'GPU_SETTING\ntarget:{target}')
    # Launch the graph

    if is_log and log_interval_epochs is None:
        log_interval_epochs = (epochs-start_epochs)//10
        if log_interval_epochs == 0:
            log_interval_epochs = 1
    if test_feed_dict is None:
        test_feed_dict = feed_dict.copy()
    test_feed_dict[x_placeholder] = test_x
    test_feed_dict[y_placeholder] = test_y
    
    total_batch = int(train_x.shape[0] / batch_size)
    
    with tf.Session(target=target, graph=graph, config=config) as sess:
        if is_log:
            merged = tf.summary.merge_all()
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)
            train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        #to close train_writer when Error occurs
        try:
            if load_model_name is None:
                sess.run(tf.global_variables_initializer())

            if is_save or is_save_last or load_model_name is not None:
                saver = tf.train.Saver(max_to_keep=max_to_keep)
                if load_model_name is not None:
                    saver.restore(sess, load_model_name)
            train_show_list_dict = {}
            train_list_dict = {}
            test_list_dict = {}
            for k in show_nodes.keys():
                train_list_dict[k] = []
                test_list_dict[k] = []

            # Training cycle
            for epoch in range(start_epochs, epochs):
                if is_print:
                    print(f'epoch={epoch}:', end=' ')
                index_arr = np.arange(train_x.shape[0])
                np.random.shuffle(index_arr)
                loss_list = []
                accuracy_list = []
                list_dict = {}
                for k in show_nodes.keys():
                    list_dict[k] = []
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = train_x[index_arr[batch_size*i:batch_size*(i+1)]]
                    batch_y = train_y[index_arr[batch_size*i:batch_size*(i+1)]]
                    tmp_feed_dict = feed_dict.copy()
                    tmp_feed_dict[x_placeholder] = batch_x
                    tmp_feed_dict[y_placeholder] = batch_y
                    calc_dict = {'opt':optimizer_node}
                    for k in show_nodes.keys():
                        calc_dict[k] = show_nodes[k]
                    d = sess.run(calc_dict, feed_dict=tmp_feed_dict)
                    for k in show_nodes.keys():
                        list_dict[k].append(d[k])
                if is_print:
                    print(f'train: time={time.time()-start_time:.4},', end=' ')
                for k in show_nodes.keys():
                    train_list_dict[k].append(list_dict[k])
                    if is_print:
                        print(f'{k}={np.mean(list_dict[k]):},', end=' ')

                #calculation with test data
                if test_x is not None:
                    if is_print:
                        print('test:', end=' ')
                    calc_dict = {}
                    if is_log:
                        if epoch%log_interval_epochs==0 and merged is not None:
                            calc_dict['__summary__'] = merged
                    for k in show_nodes.keys():
                        calc_dict[k] = show_nodes[k]
                    if len(calc_dict.keys()) != 0:
                        d = sess.run(calc_dict, feed_dict=test_feed_dict)
                    else:
                        d = {}
                    if '__summary__' in calc_dict.keys():
                        train_writer.add_summary(d['__summary__'], epoch)
                    for k in show_nodes.keys():
                        test_list_dict[k].append(d[k])
                        if is_print:
                            print(f'{k}={np.mean(d[k])},', end=' ')
                if is_print:
                    print()

                #save params
                if is_save and (epoch+1)%save_interval_epochs==0 and (is_save_last and (epoch==(epochs-1)))==False:
                    save_name = model_name+f'-{epoch+1}'
                    if os.path.exists(save_name):
                        os.remove(save_name)
                    saver.save(sess, save_name)
                    print('Saved the model. name is '+save_name)
            
            #training is end
            if is_save_last:
                save_name = model_name+f'-{epochs}-last'
                if os.path.exists(save_name):
                    os.remove(save_name)
                saver.save(sess, save_name)
                if is_print:
                    print('Saved the model. name is '+save_name)
                save_name = './'+model_name+f'-{epochs}'
        except:
            try:
                train_writer.close()
            except:
                pass
            raise
        if is_log:
            train_writer.close()
    
    if is_print:
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S")+'\n|----------------------Tensor Trainer End----------------------|\n')
        
    return train_list_dict, test_list_dict
