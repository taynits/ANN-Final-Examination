# 人工神经网络课程考察实验

## 01-猫咪分类
### 环境配置

Python 3.10.9

torch 2.1.2

CUDA Version: 12.0

### 使用方法

- Test
   ```
  cd 01-猫咪分类

  # 测试网络在验证集的分类结果，打印损失与分类正确率
  python CNN-Test.py 
  python Transformer-Test.py 
  python GAN-Test.py 

  # 测试网络对无标签测试集的预测结果
  python CNN-Predict.py 
  python Transformer-Predict.py  
  ```

   [CNN-Test.py](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/CNN-Test.py) / [Transformer-Test.py](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Transformer-Test.py)：加载训练好的模型权重，用于测试test数据集，
   分类结果保存在 [CNN_test_result.txt](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/CNN_test_result.txt) / [Transformer_test_result.txt](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/Transformer_test_result.txt)

   (由于模型在GPU上训练，保存的是GPU模型，若在cpu上测试需转换为CPU模型)

- Train and Eval
    ```
      cd 01-猫咪分类
      python CNN-Train.py  # 训练CNN网络
      python Transformer-Train.py  # 训练Transformer网络
      python GAN-Pretrain.py  # 训练GAN图像生成网络，为GAN分类器作准备
      python GAN-Train.py  # 训练GAN分类器网络
    ```

 - 01-猫咪分类/Logs 目录存放训练模型过程的日志
    
    [CNN.log](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/CNN.log) / [Transformer.log](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/Transformer.log) / [GAN_Classifier.log](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/GAN_Classifier.log)：记录运行输出，包括模型结构、训练和验证集上的损失、正确率

    [GAN.log](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/GAN.log)：记录训练GAN图像生成网络的损失、真图像和假图像输入生成器后输出的平均概率
    
    
    [logs_CNN](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/logs_CNN) / [logs_Transformer](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/logs_Transformer) / [logs_GAN](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/logs_GAN) / [logs_GANClassifier](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/Logs/logs_GANClassifier)：训练集和验证集上损失与正确率的可视化数据，所在目录下终端输入以下命令，点击返回的连接即可查看。

    ```
    tensorboard --logdir=logs_CNN 
    tensorboard --logdir=logs_Transformer
    tensorboard --logdir=logs_GANClassifier
    tensorboard --logdir=logs_GAN
    ```

- 01-猫咪分类/result 目录存放训练与预测结果

    [model_CNN_best.pth](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/model_CNN_best.pth) / [model_Transformer_best.pth](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/model_Transformer_best.pth) / [model_GAN_best.pth](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/model_GAN_best.pth)：分类正确率最高的模型权重

    [generator.pth](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/generator.pth) / [discriminator.pth](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/discriminator.pth) ：GAN图像生成网络的生成器与判别器权重

    [CNN_test_result.txt](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/CNN_test_result.txt) / [Transformer_test_result.txt](01-%E7%8C%AB%E5%92%AA%E5%88%86%E7%B1%BB/results/Transformer_test_result.txt) ：对无标签的test测试集的预测结果
    

## 02-新闻标题分类
### 环境配置
Python 3.10.9

torch 2.1.2

CUDA Version: 12.0

### 使用方法

- Test
   ```
  cd 02-新闻标题分类

  # 测试网络对无标签测试集test.txt的预测结果
  python CNN-Test.py 
  python Transformer-Test.py  
  ```

   [CNN-Test.py](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/CNN-Test.py) / [Transformer-Test.py](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/Transformer-Test.py)：加载训练好的模型权重，用于测试test.txt数据集，
   分类结果保存在 [results/CNN_test_result.txt ](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/results/CNN_test_result.txt) / [Transformer_test_result.txt](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/results/Transformer_test_result.txt)

   (由于模型在GPU上训练，保存的是GPU模型，若在cpu上测试需转换为CPU模型)

- Train and Eval
    ```
      cd 02-新闻标题分类
      python CNN-Train.py  # 训练CNN网络
      python Transformer-Train.py  # 训练Transformer网络
    ```

    为方便处理数据集，已将给定的train.txt/dev.txt转为traun.tsv/dev.tsv格式。

 - 02-新闻标题分类/Logs 目录存放训练模型过程的日志
    
    [CNN.log](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/Logs/CNN.log) / [Transformer.log](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/Logs/Transformer.log)：记录运行输出，包括模型结构、训练和验证集上的损失、正确率

    
    [logs_CNN](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/Logs/logs_CNN) / [logs_Transformer](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/Logs/logs_Transformer)：训练集和验证集上损失与正确率的可视化数据，所在目录下终端输入以下命令，点击返回的连接即可查看。

    ```
    tensorboard --logdir=logs_CNN 
    tensorboard --logdir=logs_Transformer
    ```

- 02-新闻标题分类/result 目录存放训练与预测结果

    [ model_CNN_best.pth](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/results/model_CNN_best.pth) / [model_TRANS_best.pth](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/results/model_TRANS_best.pth)：分类正确率最高的模型权重

    [CNN_test_result.txt](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/results/CNN_test_result.txt) / [Transformer_test_result.txt](02-%E6%96%B0%E9%97%BB%E6%A0%87%E9%A2%98%E5%88%86%E7%B1%BB/results/Transformer_test_result.txt)：对无标签的test测试集的预测结果


## 04-中英文翻译
### 环境配置

CPU 2核

Python 3.10.10

PaddlePaddle 2.6.0

paddlenlp 2.7.1

paddlepaddle 2.6.0

### 使用方法
aistudio项目地址: https://aistudio.baidu.com/projectdetail/7396701?channel=0&channelType=0&sUid=2569785&shared=1&ts=1705209160601

work目录下存放数据集train_dev_test

main.ipynb: 模型搭建、训练、验证，其中 do_predict(args) 模块用于预测，结果保存在work/train_dev_test/predict.txt，tokenize后的结果保存在work/train_dev_test/predict.tok.txt

！报错提示: 导入attrdict包时报错的解决方案已在对应位置说明