### SavedModel预估c++

离线训练.cpkt模型导出为savedmodel格式后, 使用c++API预估(模拟线上抽取特征后的打分过程)

`data/data.txt`: csv格式

`conf/schema.yaml`: rawdata的每一列的特征名

`conf/feature_config`: 该savedmodel使用特征以及数据类型, 即savedmodel中inputs的key

`model/`: 导出的SavedModel路径 

step1: Makefile编译源码通过生成predic执行文件

step2: 修改上述配置文件

step3:执行run.sh
    
