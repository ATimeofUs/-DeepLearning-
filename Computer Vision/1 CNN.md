# 训练日记

##  CNN

### Round 1

```markdown

Epoch 1/10
9s 20ms/step - accuracy: 0.4404 - loss: 1.5425 - val_accuracy: 0.5489 - val_loss: 1.2635
Epoch 2/10
8s 20ms/step - accuracy: 0.6002 - loss: 1.1371 - val_accuracy: 0.6192 - val_loss: 1.0766
Epoch 3/10
8s 20ms/step - accuracy: 0.6656 - loss: 0.9511 - val_accuracy: 0.6477 - val_loss: 1.0085
Epoch 4/10
8s 20ms/step - accuracy: 0.7107 - loss: 0.8292 - val_accuracy: 0.6867 - val_loss: 0.8862
Epoch 5/10
 8s 20ms/step - accuracy: 0.7458 - loss: 0.7298 - val_accuracy: 0.7173 - val_loss: 0.8092
Epoch 6/10
8s 20ms/step - accuracy: 0.7775 - loss: 0.6455 - val_accuracy: 0.7109 - val_loss: 0.8566
Epoch 7/10
8s 21ms/step - accuracy: 0.8033 - loss: 0.5687 - val_accuracy: 0.7319 - val_loss: 0.8017
Epoch 8/10
8s 21ms/step - accuracy: 0.8237 - loss: 0.5043 - val_accuracy: 0.7411 - val_loss: 0.7928
Epoch 9/10
8s 21ms/step - accuracy: 0.8498 - loss: 0.4294 - val_accuracy: 0.7314 - val_loss: 0.8398
Epoch 10/10
 8s 21ms/step - accuracy: 0.8693 - loss: 0.3727 - val_accuracy: 0.7416 - val_loss: 0.8730
 1s 7ms/step - accuracy: 0.7416 - loss: 0.8730
```

注意到train_ds的准确率高达**86%**，但是test_ds只有**74%**，模型发生**overfiting**

<span style="text-decoration:line-through;">其实这个是训练比较好的，这似乎就是卷积神经网络的极限</span>

### Round 2

接下来采用将图片旋转四个方向，达到把训练集放大4倍的效果，结果如下：

```txt
 33s 20ms/step - accuracy: 0.5164 - loss: 1.3423 - val_accuracy: 0.3407 - val_loss: 2.0983
Epoch 2/10
 33s 21ms/step - accuracy: 0.6388 - loss: 1.0196 - val_accuracy: 0.3854 - val_loss: 2.0146
Epoch 3/10
 35s 22ms/step - accuracy: 0.6923 - loss: 0.8739 - val_accuracy: 0.4278 - val_loss: 1.8504
Epoch 4/10
 36s 23ms/step - accuracy: 0.7262 - loss: 0.7794 - val_accuracy: 0.4347 - val_loss: 1.8766
Epoch 5/10
 35s 23ms/step - accuracy: 0.7502 - loss: 0.7108 - val_accuracy: 0.5072 - val_loss: 1.5235
Epoch 6/10
 35s 22ms/step - accuracy: 0.7705 - loss: 0.6498 - val_accuracy: 0.5138 - val_loss: 1.5742
Epoch 7/10
 33s 21ms/step - accuracy: 0.7888 - loss: 0.5974 - val_accuracy: 0.4621 - val_loss: 1.9706
Epoch 8/10
 34s 22ms/step - accuracy: 0.8041 - loss: 0.5559 - val_accuracy: 0.5009 - val_loss: 1.7622
Epoch 9/10
 34s 21ms/step - accuracy: 0.8198 - loss: 0.5099 - val_accuracy: 0.5142 - val_loss: 1.6746
Epoch 10/10
 32s 21ms/step - accuracy: 0.8351 - loss: 0.4675 - val_accuracy: 0.5175 - val_loss: 1.7547
 1s 7ms/step - accuracy: 0.5175 - loss: 1.7547

测试集上的准确率: 0.5175
测试集上的损失: 1.7547
```

过拟合更加严重了！

### Round 3

我去掉旋转180度的方法，并且将 MaxPool2D改为 AvgPool2D，看看效果会不会更好

<span style="text-decoration:line-through;">没有，一坨屎，test才50%，就不展示了</span>



### Round 4

问了问AI，以下是修复方案：

1. 移除 load_data 中的 np.rot90 代码。
2. 添加 Keras 内置的数据增强层（水平翻转、小角度旋转）。
3. 添加 Dropout 层丢弃部分神经元，防止过拟合。
4. 添加 BatchNormalization 加速收敛并稳定模型。（归一化层，有两个学习参数）
5. 将 AvgPool2D 改为 MaxPooling2D（通常在 CNN 中表现更好）。

```

Epoch 1/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 53s 67ms/step - accuracy: 0.3780 - loss: 1.7312 - val_accuracy: 0.5047 - val_loss: 1.3789
Epoch 2/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 50s 64ms/step - accuracy: 0.5031 - loss: 1.3946 - val_accuracy: 0.5863 - val_loss: 1.1491
Epoch 3/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 52s 66ms/step - accuracy: 0.5641 - loss: 1.2416 - val_accuracy: 0.6377 - val_loss: 1.0324
Epoch 4/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 49s 63ms/step - accuracy: 0.6056 - loss: 1.1361 - val_accuracy: 0.6413 - val_loss: 1.0629
Epoch 5/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 50s 63ms/step - accuracy: 0.6343 - loss: 1.0647 - val_accuracy: 0.5863 - val_loss: 1.2700
Epoch 6/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 53s 68ms/step - accuracy: 0.6538 - loss: 1.0091 - val_accuracy: 0.6385 - val_loss: 1.0546
Epoch 7/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 51s 65ms/step - accuracy: 0.6727 - loss: 0.9631 - val_accuracy: 0.6017 - val_loss: 1.2041
Epoch 8/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 51s 66ms/step - accuracy: 0.6860 - loss: 0.9187 - val_accuracy: 0.7176 - val_loss: 0.8182
Epoch 9/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 51s 65ms/step - accuracy: 0.6990 - loss: 0.8842 - val_accuracy: 0.6488 - val_loss: 1.0662
Epoch 10/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 50s 63ms/step - accuracy: 0.7132 - loss: 0.8491 - val_accuracy: 0.6939 - val_loss: 0.9147
79/79 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6939 - loss: 0.9147

```



